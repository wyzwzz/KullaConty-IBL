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
        auto load_model = [&](const std::string& name,const PBRMaterial& material,const mat4& t ){
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
            m.model = t;
            m.albedo.initialize_handle();
            m.albedo.initialize_format_and_data(12,GL_RGBA8,albedo_rsc);
            m.albedo.generate_mipmap();
            m.ao.initialize_handle();
            m.ao.initialize_format_and_data(12,GL_RGBA8,ao_rsc);
            m.ao.generate_mipmap();
            m.metal.initialize_handle();
            m.metal.initialize_format_and_data(12,GL_RGBA8,metal_rsc);
            m.metal.generate_mipmap();
            m.normal.initialize_handle();
            m.normal.initialize_format_and_data(12,GL_RGBA8,normal_rsc);
            m.normal.generate_mipmap();
            m.roughness.initialize_handle();
            m.roughness.initialize_format_and_data(12,GL_RGBA8,roughness_rsc);
            m.roughness.generate_mipmap();
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
        },transform::translate(0,0,2) * transform::rotate_y(wzz::math::deg2rad(-90.f)));

        load_model("asset/statue/statue_mesh.obj",{
                .albedo = "asset/statue/marble_albedo.png",
                .ao = "asset/statue/marble_ao.png",
                .metal = "asset/statue/marble_metal.png",
                .normal = "asset/statue/marble_normal.png",
                .roughness = "asset/statue/marble_rough.png"
        },transform::rotate_x(wzz::math::deg2rad(90.f)));

        load_model("asset/DamagedHelmet/DamagedHelmet.obj",
                   {
            .albedo = "asset/DamagedHelmet/Default_albedo.jpg",
            .ao = "asset/DamagedHelmet/Default_AO.jpg",
            .metal = "asset/DamagedHelmet/metallic.png",
            .normal = "asset/DamagedHelmet/Default_normal.jpg",
            .roughness = "asset/DamagedHelmet/roughness.png"
            },transform::translate(vec3(3,0,0)));

        ibl_shader = program_t::build_from(
                shader_t<GL_VERTEX_SHADER>::from_file("asset/glsl/ibl.vert"),
                shader_t<GL_FRAGMENT_SHADER>::from_file("asset/glsl/ibl.frag")
                );
        ibl_params_buffer.initialize_handle();
        ibl_params_buffer.reinitialize_buffer_data(nullptr,GL_STATIC_DRAW);


        diffuse_env.initialize_handle();
        diffuse_env.initialize_texture(1,GL_RGBA32F,diffuse_env_size.x,diffuse_env_size.y);

        diffuse_shader = program_t::build_from(
                shader_t<GL_COMPUTE_SHADER>::from_file("asset/glsl/diffuse.comp")
                );

        linear_sampler.initialize_handle();
        linear_sampler.set_param(GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
        linear_sampler.set_param(GL_TEXTURE_MAG_FILTER,GL_LINEAR);

        lut_sampler.initialize_handle();
        lut_sampler.set_param(GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
        lut_sampler.set_param(GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
        lut_sampler.set_param(GL_TEXTURE_MAG_FILTER,GL_LINEAR);

        light_shader = program_t::build_from(
                shader_t<GL_COMPUTE_SHADER>::from_file("asset/glsl/light.comp")
                );
        light_env.initialize_handle();
        light_env.initialize_texture(max_mip_level,GL_RGBA32F,light_env_size.x,light_env_size.y);
        light_params_buffer.initialize_handle();
        light_params_buffer.reinitialize_buffer_data(nullptr,GL_STATIC_DRAW);

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


        auto sphere = MakeSphere();
        test_ball.vao.initialize_handle();
        test_ball.pos.initialize_handle();
        test_ball.normal.initialize_handle();
        test_ball.texcoord.initialize_handle();
        test_ball.ebo.initialize_handle();
        test_ball.pos.reinitialize_buffer_data(sphere.positions.data(), sphere.positions.size(), GL_STATIC_DRAW);
        test_ball.normal.reinitialize_buffer_data(sphere.normals.data(),sphere.normals.size(),GL_STATIC_DRAW);
        test_ball.texcoord.reinitialize_buffer_data(sphere.uv.data(),sphere.uv.size(),GL_STATIC_DRAW);
        test_ball.vao.bind_vertex_buffer_to_attrib(attrib_var_t<vec3f>(0), test_ball.pos, 0);
        test_ball.vao.enable_attrib(attrib_var_t<vec3f>(0));
        test_ball.vao.bind_vertex_buffer_to_attrib(attrib_var_t<vec3f>(1),test_ball.normal,1);
        test_ball.vao.enable_attrib(attrib_var_t<vec3f>(1));
        test_ball.vao.bind_vertex_buffer_to_attrib(attrib_var_t<vec2f>(2),test_ball.texcoord,2);
        test_ball.vao.enable_attrib(attrib_var_t<vec2f>(2));
        test_ball.ebo.reinitialize_buffer_data(sphere.indices.data(), sphere.indices.size(), GL_STATIC_DRAW);
        test_ball.vao.bind_index_buffer(test_ball.ebo);
        pbr_test_params_buffer.initialize_handle();
        pbr_test_params_buffer.reinitialize_buffer_data(&pbr_test_params,GL_STATIC_DRAW);
        ibl_test_shader = program_t::build_from(
                shader_t<GL_VERTEX_SHADER>::from_file("asset/glsl/ibl.vert"),
                shader_t<GL_FRAGMENT_SHADER>::from_file("asset/glsl/ibl_test.frag")
                );
        env_name_path_mp["outside day"] = "asset/hdr/environment_day.hdr";
        env_name_path_mp["outside dusk"] = "asset/hdr/environment_dusk.hdr";
        env_name_path_mp["small hangar"] = "asset/hdr/small_hangar_01_4k.hdr";
        cur_env_name = "outside day";
        loadEnvMap(env_name_path_mp[cur_env_name],cur_env_name);
        precompute(true,true, false, false);
    }

    void frame() override{
        handle_events();
        if(ImGui::Begin("Settings",nullptr,ImGuiWindowFlags_AlwaysAutoResize)){
            ImGui::Text("Press LCtrl to show/hide cursor");
            ImGui::Text("Use W/A/S/D/Space/LShift to move");
            ImGui::Text("FPS: %.0f", ImGui::GetIO().Framerate);
            if (ImGui::Checkbox("VSync", &vsync))
                window->set_vsync(vsync);

            static const std::string env_lights[] = {
                    "outside day",
                    "outside dusk",
                    "small hangar"
            };
            if(ImGui::BeginCombo("EnvLight Map",cur_env_name.c_str())){
                for(int i = 0; i < 3; i++){
                    bool select = env_lights[i] == cur_env_name;
                    if(ImGui::Selectable(env_lights[i].c_str(),select)){
                        cur_env_name = env_lights[i];
                        loadEnvMap(env_name_path_mp[cur_env_name],cur_env_name);
                    }
                }
                ImGui::EndCombo();
            }

            ImGui::Checkbox("Enable KC",&enable_kc);
            if(enable_kc){
                ImGui::ColorEdit3("Edge Tint",&edge_tint.x);
            }

            ImGui::Checkbox("Show Test Ball",&show_test);
            if(show_test){
                bool update = false;
                ImGui::InputFloat3("Ball Center Pos",&test_ball.center_pos.x);
                update |= ImGui::ColorEdit3("Ball Albedo",&pbr_test_params.albedo.x);
                update |= ImGui::ColorEdit3("Ball Edge Tint",&pbr_test_params.edge_tint.x);
                update |= ImGui::SliderFloat("Ball Roughness",&pbr_test_params.roughness,0,1);
                update |= ImGui::SliderFloat("Ball Metallic",&pbr_test_params.metallic,0,1);
                if(update){
                    pbr_test_params_buffer.set_buffer_data(&pbr_test_params);
                }
            }
        }
        ImGui::End();

        framebuffer_t::clear_color_depth_buffer();





        diffuse_env.bind(5);
        light_env.bind(6);
        brdf.bind(7);
        emiu.bind(8);
        eavg.bind(9);
        linear_sampler.bind(5);
        linear_sampler.bind(6);
        lut_sampler.bind(7);
        lut_sampler.bind(8);
        lut_sampler.bind(9);
        ibl_params.camera_pos = camera.get_position();
        ibl_params.max_refl_lod = max_mip_level - 1;
        ibl_params.edge_tint = edge_tint;
        ibl_params.enable_kc = static_cast<int>(enable_kc);
        ibl_params_buffer.set_buffer_data(&ibl_params);
        ibl_params_buffer.bind(10);

        if(show_test){
            ibl_test_shader.bind();
            auto model = transform::translate(test_ball.center_pos);
            ibl_test_shader.set_uniform_var("Model",model);
            ibl_test_shader.set_uniform_var("ProjView",camera.get_view_proj());
            pbr_test_params_buffer.bind(11);

            test_ball.vao.bind();

            GL_EXPR(glDrawElements(GL_TRIANGLE_STRIP,test_ball.ebo.index_count(),GL_UNSIGNED_INT,nullptr));

            test_ball.vao.unbind();
            ibl_test_shader.unbind();
        }


        ibl_shader.bind();
        ibl_shader.set_uniform_var("ProjView",camera.get_view_proj());
        for(auto& draw_model:draw_models){
            ibl_shader.set_uniform_var("Model",draw_model.model);

            draw_model.albedo.bind(0);
            draw_model.normal.bind(1);
            draw_model.metal.bind(2);
            draw_model.roughness.bind(3);
            draw_model.ao.bind(4);
            linear_sampler.bind(0);
            linear_sampler.bind(1);
            linear_sampler.bind(2);
            linear_sampler.bind(3);
            linear_sampler.bind(4);

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
    void loadEnvMap(const std::string& path,const std::string& name){
        if(env_mp.count(name) == 0)
            env_mp[name] = decltype(env_mp)::value_type::second_type(load_rgb_from_hdr_file(path)).flip_vertically();

        env_map.destroy();
        env_map.initialize_handle();
        auto res = env_mp[name].size();
        auto x = std::max(res.x,res.y);
        env_map.initialize_format_and_data(log2(x),GL_RGB32F,env_mp[name]);
        env_map.generate_mipmap();
        precompute(false,false,true,true);
        cur_env_name = name;
    }
    void precompute(bool kc = true,bool lut = true,bool diffuse = true,bool light = true) {
        if(kc){
            auto Emiu = generateEmiu(lut_size, sample_count);
            auto Eavg = generateEavg(Emiu);
            emiu.set_texture_data(lut_size.x, lut_size.y, Emiu.get_raw_data());
            eavg.set_texture_data(lut_size.x, Eavg.get_raw_data());
        }
        if(lut){
            brdf_int_shader.bind();
            auto group_size = getGroupSize(lut_size.x, lut_size.y);
            sample_params_buffer.bind(0);
            emiu.bind(0);
            eavg.bind(1);
            lut_sampler.bind(0);
            lut_sampler.bind(1);
            brdf.bind_image(2, 0, GL_WRITE_ONLY, GL_RGBA32F);
            GL_EXPR(glDispatchCompute(group_size.x, group_size.y, group_size.z));
            brdf_int_shader.unbind();
        }
        if(diffuse){
            diffuse_shader.bind();
            auto group_size = getGroupSize(diffuse_env_size.x, diffuse_env_size.y);
            sample_params_buffer.bind(0);
            env_map.bind(0);
            linear_sampler.bind(0);
            diffuse_env.bind_image(1,0,GL_WRITE_ONLY,GL_RGBA32F);
            GL_EXPR(glDispatchCompute(group_size.x, group_size.y, group_size.z));
            diffuse_shader.unbind();
        }
        if(light){
            light_shader.bind();
            sample_params_buffer.bind(0);
            light_params_buffer.bind(2);
            env_map.bind(0);
            linear_sampler.bind(0);
            for(int mip = 0; mip < max_mip_level; ++mip){
                light_env.bind_image(1,mip,GL_WRITE_ONLY,GL_RGBA32F);
                light_params.roughness = static_cast<float>(mip) / (max_mip_level - 1.f);
                light_params_buffer.set_buffer_data(&light_params);
                vec2i mip_light_env_size = light_env_size / static_cast<int>(pow(2,mip));
                auto group_size = getGroupSize(mip_light_env_size.x,mip_light_env_size.y);
                GL_EXPR(glDispatchCompute(group_size.x,group_size.y,group_size.z));
            }

            light_shader.unbind();
        }
    }
    private:
    bool vsync = true;
    const vec2i lut_size = vec2i(512,512);
    const int sample_count = 1024;
    program_t brdf_int_shader;
    vec3 edge_tint = vec3(1);
    struct alignas(16) SampleParams{
        int sample_count = 1024;
    }sample_params;
    std140_uniform_block_buffer_t<SampleParams> sample_params_buffer;

    texture2d_t emiu;
    texture1d_t eavg;
    texture2d_t brdf;

    sampler_t linear_sampler;
    sampler_t lut_sampler;

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
        texture2d_t emissive;
        mat4 model;
        std::vector<DrawMesh> draw_meshes;
        std::string name;
    };
    std::vector<PBRModel> draw_models;

    program_t ibl_shader;
    bool enable_kc = true;
    struct alignas(16) IBLParams{
        vec3 camera_pos;
        float max_refl_lod;
        vec3 edge_tint;
        int enable_kc = static_cast<int>(enable_kc);
    }ibl_params;
    std140_uniform_block_buffer_t<IBLParams> ibl_params_buffer;

    texture2d_t env_map;
    std::string cur_env_name;
    std::unordered_map<std::string,std::string> env_name_path_mp;
    std::unordered_map<std::string,wzz::texture::image2d_t<wzz::math::color3f>> env_mp;

    program_t diffuse_shader;
    texture2d_t diffuse_env;
    vec2i diffuse_env_size = {64,64};

    program_t light_shader;
    texture2d_t light_env;
    vec2i light_env_size = {1024,512};
    int max_mip_level = 5;
    struct alignas(16) LightParams{
        float roughness;
    }light_params;
    std140_uniform_block_buffer_t<LightParams> light_params_buffer;

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

    struct Ball{
        vertex_array_t vao;
        vertex_buffer_t<vec3f> pos;
        vertex_buffer_t<vec3f> normal;
        vertex_buffer_t<vec2f> texcoord;
        index_buffer_t<uint32_t> ebo;
        vec3 center_pos;
    }test_ball;
    bool show_test = true;
    struct PBRTest{
        vec3 albedo = vec3(1);
        float roughness = 0;
        vec3 edge_tint = vec3(1);
        float metallic = 1;
    }pbr_test_params;
    std140_uniform_block_buffer_t<PBRTest> pbr_test_params_buffer;
    program_t ibl_test_shader;
};

int main(){
    KullaContyIBL(window_desc_t{
        .size = {1920,1080},
        .title = "KullaContyIBL",
        .resizeable = false,
        .multisamples = 4,
    }).run();
}