#include "common.hpp"

#ifndef NDEBUG
#define set_uniform_var set_uniform_var_unchecked
#endif

inline vec3i getGroupSize(int x, int y = 1, int z = 1) {
    constexpr int group_thread_size_x = 16;
    constexpr int group_thread_size_y = 16;
    constexpr int group_thread_size_z = 16;
    const int group_size_x = (x + group_thread_size_x - 1) / group_thread_size_x;
    const int group_size_y = (y + group_thread_size_y - 1) / group_thread_size_y;
    const int group_size_z = (z + group_thread_size_z - 1) / group_thread_size_z;
    return {group_size_x, group_size_y, group_size_z};
}

float geometrySchlickGGX(float NdotV, float roughness){
    float a = roughness;
    float k = (a * a) / 2.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

float geometrySmith(float NdotV,float NdotL, float roughness){
    float ggx2 = geometrySchlickGGX(NdotV,roughness);
    float ggx1 = geometrySchlickGGX(NdotL,roughness);

    return ggx1 * ggx2;
}


vec2 hammersley(uint32_t i, uint32_t N)
{
    uint32_t bits = (i << 16u) | (i >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    float rdi = static_cast<float>(bits * 2.3283064365386963e-10);
    return { static_cast<float>(i) / N, rdi };
}

vec3 sampleGGX(float roughness, float u1, float u2)
{
    float alpha = roughness * roughness;
    float theta = std::atan(alpha * std::sqrt(u1) / std::sqrt(1 - u1));
    float phi   = 2 * PI * u2;

    return {std::sin(phi) * std::sin(theta),
            std::cos(phi) * std::sin(theta),
            std::cos(theta)};
}


auto generateEmiu(const vec2i& res,int spp){
    wzz::texture::image2d_t<float> Emiu(res.x,res.y);
    parallel_forrange(0,res.y,[&](int,int y){
        const float roughness = (y + 0.5f) / res.y;
        for(int x = 0; x < res.x; ++x){
            const float NdotV = (x + 0.5f) / res.x;
            const vec3f wo = vec3f(std::sqrt(std::max(0.f,1.f - NdotV * NdotV)),0,NdotV);
            float sum = 0;
            for(int i = 0; i < spp; ++i){
                const vec2 xi = hammersley(i,spp);
                const vec3 wh = sampleGGX(roughness,xi.x,xi.y);
                const vec3 wi = (2.f * wh * dot(wh,wo) - wo).normalized();
                if(wi.z <= 0) continue;
                const float G = geometrySmith(wi.z,wo.z,roughness);
                float weight = dot(wo,wh) * G / (wo.z * wh.z);
                if(std::isfinite(weight))
                    sum += weight;
            }
            Emiu.at(x,y) = std::min(sum / spp, 1.f);
        }
    },20);
    return Emiu;
}

auto generateEavg(const wzz::texture::image2d_t<float>& Emiu){
    wzz::texture::image1d_t<float> Eavg(Emiu.height());
    const float dmu = 1.f / Emiu.width();
    for(int y = 0; y < Emiu.height(); ++y){
        float sum = 0;
        for(int x = 0; x < Emiu.width(); ++x){
            const float miu = (x + 0.5f) / Emiu.width();
            sum += miu * Emiu(x,y) * dmu;
        }
        Eavg(y) = 2 * sum;
    }
    return Eavg;
}

class KullaContyIBL:public gl_app_t{
public:
    using gl_app_t::gl_app_t;
private:
    void initialize() override{
        GL_EXPR(glEnable(GL_DEPTH_TEST));
        GL_EXPR(glClearColor(0,0,0,0));

        brdf_int_shader = program_t::build_from(
                shader_t<GL_COMPUTE_SHADER>::from_file("asset/glsl/brdf.comp")
                );

        sample_params_buffer.initialize_handle();
        sample_params_buffer.reinitialize_buffer_data(&sample_params,GL_STATIC_DRAW);

        emiu.initialize_handle();
        emiu.initialize_texture(1,GL_R32F,lut_size.x,lut_size.y);

        eavg.initialize_handle();
        eavg.initialize_texture(1,GL_R32F,lut_size.x);

        brdf.initialize_handle();
        brdf.initialize_texture(1,GL_RGBA32F,lut_size.x,lut_size.y);

        //load model
        struct PBRMaterial{
            std::string albedo;
            std::string ao;
            std::string metal;
            std::string normal;
            std::string roughness;
        };
        auto load_model = [&](const std::string& name,const PBRMaterial& material){
            auto model = load_model_from_obj_file(name);
            LOG_INFO("load mesh ok");
            draw_models.emplace_back();
            auto& m = draw_models.back();
            auto albedo_rsc = wzz::texture::image2d_t(load_rgba_from_file(material.albedo)).flip_vertically();
            auto ao_rsc = wzz::texture::image2d_t(load_rgba_from_file(material.ao)).flip_vertically();
            auto metal_rsc = wzz::texture::image2d_t(load_rgba_from_file(material.metal)).flip_vertically();
            auto normal_rsc = wzz::texture::image2d_t(load_rgba_from_file(material.normal)).flip_vertically();
            auto roughness_rsc = wzz::texture::image2d_t(load_rgba_from_file(material.roughness)).flip_vertically();
            LOG_INFO("load texture ok");
            m.albedo.initialize_handle();
            m.albedo.initialize_format_and_data(1,GL_RGBA8,albedo_rsc);
            m.ao.initialize_handle();
            m.ao.initialize_format_and_data(1,GL_RGBA8,ao_rsc);
            m.metal.initialize_handle();
            m.metal.initialize_format_and_data(1,GL_RGBA8,metal_rsc);
            m.normal.initialize_handle();
            m.normal.initialize_format_and_data(1,GL_RGBA8,normal_rsc);
            m.roughness.initialize_handle();
            m.roughness.initialize_format_and_data(1,GL_RGBA8,roughness_rsc);
            for(auto& mesh:model->meshes){
                m.draw_meshes.emplace_back();
                auto& mm = m.draw_meshes.back();
                mm.vao.initialize_handle();
                mm.vbo.initialize_handle();
                mm.ebo.initialize_handle();
                mm.vbo.reinitialize_buffer_data(mesh.vertices.data(),mesh.vertices.size(),GL_STATIC_DRAW);
                mm.vao.bind_vertex_buffer_to_attrib(attrib_var_t<vec3f>(0),mm.vbo,&vertex_t::pos,0);
                mm.vao.bind_vertex_buffer_to_attrib(attrib_var_t<vec3f>(1),mm.vbo,&vertex_t::normal,1);
                mm.vao.bind_vertex_buffer_to_attrib(attrib_var_t<vec2f>(2),mm.vbo,&vertex_t::tex_coord,2);
                mm.vao.enable_attrib(attrib_var_t<vec3f>(0));
                mm.vao.enable_attrib(attrib_var_t<vec3f>(1));
                mm.vao.enable_attrib(attrib_var_t<vec3f>(2));
                mm.ebo.reinitialize_buffer_data(mesh.indices.data(),mesh.indices.size(),GL_STATIC_DRAW);
                mm.vao.bind_index_buffer(mm.ebo);
            }
            LOG_INFO("load {} ok",name);
        };
        load_model("asset/cerberus/cerberus_mesh.obj",{
            .albedo = "asset/cerberus/cerberus_albedo.png",
            .ao = "asset/cerberus/cerberus_ao.png",
            .metal = "asset/cerberus/cerberus_metal.png",
            .normal = "asset/cerberus/cerberus_normal.png",
            .roughness = "asset/cerberus/cerberus_rough.png"
        });

        ibl_shader = program_t::build_from(
                shader_t<GL_VERTEX_SHADER>::from_file("asset/glsl/ibl.vert"),
                shader_t<GL_FRAGMENT_SHADER>::from_file("asset/glsl/ibl.frag")
                );

        env_map.initialize_handle();
        env_map.initialize_format_and_data(
                1,GL_RGB32F,
                wzz::texture::image2d_t(wzz::image::load_rgb_from_hdr_file("asset/hdr/environment_dusk.hdr")).flip_vertically());

        diffuse_env.initialize_handle();
        diffuse_env.initialize_texture(1,GL_RGBA32F,diffuse_env_size.x,diffuse_env_size.y);

        diffuse_shader = program_t::build_from(
                shader_t<GL_COMPUTE_SHADER>::from_file("asset/glsl/diffuse.comp")
                );

        linear_sampler.initialize_handle();
        linear_sampler.set_param(GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
        linear_sampler.set_param(GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
        linear_sampler.set_param(GL_TEXTURE_WRAP_R,GL_CLAMP_TO_EDGE);
        linear_sampler.set_param(GL_TEXTURE_MIN_FILTER,GL_LINEAR);
        linear_sampler.set_param(GL_TEXTURE_MAG_FILTER,GL_LINEAR);

        //skybox
        vec3f vertices[] = {
                // back face
                {-1.0f, -1.0f, -1.0f},{1.0f, 1.0f, -1.0f},{1.0f, -1.0f, -1.0f},
                {1.0f, 1.0f, -1.0f},{-1.0f, -1.0f, -1.0f},{-1.0f, 1.0f, -1.0f},
                // front face
                {-1.0f, -1.0f, 1.0f},{1.0f, -1.0f, 1.0f},{1.0f, 1.0f, 1.0f},
                {1.0f, 1.0f, 1.0f},{-1.0f, 1.0f, 1.0f},{-1.0f, -1.0f, 1.0f},
                // left face
                {-1.0f, 1.0f, 1.0f},{-1.0f, 1.0f, -1.0f},{-1.0f, -1.0f, -1.0f},
                {-1.0f, -1.0f, -1.0f},{-1.0f, -1.0f, 1.0f},{-1.0f, 1.0f, 1.0f},
                // right face
                {1.0f, 1.0f, 1.0f},{1.0f, -1.0f, -1.0f},{1.0f, 1.0f, -1.0f},
                {1.0f, -1.0f, -1.0f},{1.0f, 1.0f, 1.0f},{1.0f, -1.0f, 1.0f},
                // bottom face
                {-1.0f, -1.0f, -1.0f},{1.0f, -1.0f, -1.0f},{1.0f, -1.0f, 1.0f},
                {1.0f, -1.0f, 1.0f},{-1.0f, -1.0f, 1.0f},{-1.0f, -1.0f, -1.0f},
                // top face
                {-1.0f, 1.0f, -1.0f},{1.0f, 1.0f, 1.0f},{1.0f, 1.0f, -1.0f},
                {1.0f, 1.0f, 1.0f},{-1.0f, 1.0f, -1.0f},{-1.0f, 1.0f, 1.0f},
        };
        skybox.vao.initialize_handle();
        skybox.vbo.initialize_handle();
        skybox.vbo.reinitialize_buffer_data(vertices,36,GL_STATIC_DRAW);
        skybox.vao.bind_vertex_buffer_to_attrib(attrib_var_t<vec3f>(0),skybox.vbo,0);
        skybox.vao.enable_attrib(attrib_var_t<vec3f>(0));

        skybox_shader = program_t::build_from(
                shader_t<GL_VERTEX_SHADER>::from_file("asset/glsl/skybox.vert"),
                shader_t<GL_FRAGMENT_SHADER>::from_file("asset/glsl/skybox.frag")
                );

        transform_buffer.initialize_handle();
        transform_buffer.reinitialize_buffer_data(&transform,GL_STATIC_DRAW);


        camera.set_position({0,0,5});
        camera.set_perspective(60,0.1,20);
        camera.set_direction(-wzz::math::PI_f / 2,0);

        precompute();
    }

    void frame() override{
        handle_events();
        if(ImGui::Begin("Settings",nullptr,ImGuiWindowFlags_AlwaysAutoResize)){

        }
        ImGui::End();

        framebuffer_t::clear_color_depth_buffer();


        ibl_shader.bind();
        ibl_shader.set_uniform_var("ProjView",camera.get_view_proj());
        diffuse_env.bind(5);
        brdf.bind(7);
        emiu.bind(8);
        eavg.bind(9);
        linear_sampler.bind(5);
        for(auto& draw_model:draw_models){
            ibl_shader.set_uniform_var("Model",draw_model.model);

            draw_model.albedo.bind(0);

            for(auto& draw_mesh:draw_model.draw_meshes){
                draw_mesh.vao.bind();
                GL_EXPR(glDrawElements(GL_TRIANGLES,draw_mesh.ebo.index_count(),GL_UNSIGNED_INT,nullptr));

                draw_mesh.vao.unbind();
            }

        }
        ibl_shader.unbind();

        skybox_shader.bind();
        skybox.vao.bind();
        env_map.bind(1);
        linear_sampler.bind(1);
        transform.view = camera.get_view();
        transform.proj = camera.get_proj();
        transform_buffer.set_buffer_data(&transform);
        transform_buffer.bind(0);
        GL_EXPR(glDepthFunc(GL_LEQUAL));
        GL_EXPR(glDrawArrays(GL_TRIANGLES,0,36));
        GL_EXPR(glDepthFunc(GL_LESS));
        skybox.vao.unbind();
        skybox_shader.unbind();


    }

    void destroy() override{

    }
private:
    void precompute() {
        auto Emiu = generateEmiu(lut_size,sample_count);
        auto Eavg = generateEavg(Emiu);
        emiu.set_texture_data(lut_size.x,lut_size.y,Emiu.get_raw_data());
        eavg.set_texture_data(lut_size.x,Eavg.get_raw_data());

        {
            brdf_int_shader.bind();
            auto group_size = getGroupSize(lut_size.x, lut_size.y);
            sample_params_buffer.bind(0);
            emiu.bind(0);
            eavg.bind(1);
            linear_sampler.bind(0);
            linear_sampler.bind(1);
            brdf.bind_image(2, 0, GL_WRITE_ONLY, GL_RGBA32F);
            GL_EXPR(glDispatchCompute(group_size.x, group_size.y, group_size.z));
            brdf_int_shader.unbind();
        }
        {
            diffuse_shader.bind();
            auto group_size = getGroupSize(diffuse_env_size.x, diffuse_env_size.y);
            sample_params_buffer.bind(0);
            env_map.bind(0);
            linear_sampler.bind(0);
            diffuse_env.bind_image(1,0,GL_WRITE_ONLY,GL_RGBA32F);
            GL_EXPR(glDispatchCompute(group_size.x, group_size.y, group_size.z));
            diffuse_shader.unbind();
        }
    }
    private:
    const vec2i lut_size = vec2i(512,512);
    const int sample_count = 1024;
    program_t brdf_int_shader;

    struct alignas(16) SampleParams{
        int sample_count = 1024;
    }sample_params;
    std140_uniform_block_buffer_t<SampleParams> sample_params_buffer;

    texture2d_t emiu;
    texture1d_t eavg;
    texture2d_t brdf;

    sampler_t linear_sampler;


    struct DrawMesh{
        vertex_array_t vao;
        vertex_buffer_t<vertex_t> vbo;
        index_buffer_t<uint32_t> ebo;
    };
    struct PBRModel{
        texture2d_t albedo;
        texture2d_t ao;
        texture2d_t metal;
        texture2d_t normal;
        texture2d_t roughness;
        mat4 model;
        std::vector<DrawMesh> draw_meshes;
        std::string name;
    };
    std::vector<PBRModel> draw_models;

    program_t ibl_shader;

    texture2d_t env_map;
    program_t diffuse_shader;
    texture2d_t diffuse_env;
    vec2i diffuse_env_size = {64,64};

    program_t light_shader;

    program_t skybox_shader;
    struct{
        vertex_array_t vao;
        vertex_buffer_t<vec3f> vbo;
    }skybox;
    struct Transform{
        mat4 model = mat4::identity();
        mat4 view;
        mat4 proj;
    }transform;
    std140_uniform_block_buffer_t<Transform> transform_buffer;

    program_t post_shader;
};

int main(){
    KullaContyIBL(window_desc_t{
        .size = {1280,720},
        .title = "KullaContyIBL",
        .resizeable = false,
        .multisamples = 4,
    }).run();
}