#pragma once
#include <CGUtils/api.hpp>
#include <CGUtils/model.hpp>
#include <CGUtils/image.hpp>

using namespace wzz::gl;

using namespace wzz::model;

using namespace wzz::image;

constexpr float PI = wzz::math::PI_f;

constexpr float invPI = wzz::math::invPI<float>;

template<typename T, typename Func>
void parallel_forrange(T beg, T end, Func &&func, int worker_count = 0)
{
    std::mutex it_mutex;
    T it = beg;
    auto next_item = [&]() -> std::optional<T>
    {
        std::lock_guard lk(it_mutex);
        if(it == end)
            return std::nullopt;
        return std::make_optional(it++);
    };

    std::mutex except_mutex;
    std::exception_ptr except_ptr = nullptr;

    auto worker_func = [&](int thread_index)
    {
        for(;;)
        {
            auto item = next_item();
            if(!item)
                break;

            try
            {
                func(thread_index, *item);
            }
            catch(...)
            {
                std::lock_guard lk(except_mutex);
                if(!except_ptr)
                    except_ptr = std::current_exception();
            }

            std::lock_guard lk(except_mutex);
            if(except_ptr)
                break;
        }
    };

    std::vector<std::thread> workers;
    for(int i = 0; i < worker_count; ++i)
        workers.emplace_back(worker_func, i);

    for(auto &w : workers)
        w.join();

    if(except_ptr)
        std::rethrow_exception(except_ptr);
}

struct Sphere {
    std::vector<vec3f> positions;
    std::vector<vec3f> normals;
    std::vector<vec2f> uv;
    std::vector<uint32_t> indices;
};

Sphere MakeSphere(int segcount = 64) {
    uint32_t X_SEGMENTS = 64;
    uint32_t Y_SEGMENTS = 64;

    Sphere sphere;

    for (uint32_t y = 0; y <= Y_SEGMENTS; y++) {
        for (uint32_t x = 0; x <= X_SEGMENTS; x++) {
            float x_segment = static_cast<float>(x) / X_SEGMENTS;
            float y_segment = static_cast<float>(y) / Y_SEGMENTS;
            float x_pos = std::cos(x_segment * 2.f * PI) * std::sin(y_segment * PI);
            float y_pos = std::cos(y_segment * PI);
            float z_pos = std::sin(x_segment * 2.f * PI) * std::sin(y_segment * PI);

            sphere.positions.emplace_back(x_pos, y_pos, z_pos);
            sphere.uv.emplace_back(x_segment, y_segment);
            sphere.normals.emplace_back(x_pos, y_pos, z_pos);
        }
    }
    //use GL_TRIANGLE_STRIP
    bool odd_row = false;
    for (uint32_t y = 0; y < Y_SEGMENTS; y++) {
        if (!odd_row) {
            for (uint32_t x = 0; x <= X_SEGMENTS; x++) {
                sphere.indices.emplace_back(y * (X_SEGMENTS + 1) + x);
                sphere.indices.emplace_back((y + 1) * (X_SEGMENTS + 1) + x);
            }
        } else {
            for (int x = X_SEGMENTS; x >= 0; x--) {
                sphere.indices.emplace_back((y + 1) * (X_SEGMENTS + 1) + x);
                sphere.indices.emplace_back(y * (X_SEGMENTS + 1) + x);
            }
        }
        odd_row = !odd_row;
    }

    return sphere;
}
