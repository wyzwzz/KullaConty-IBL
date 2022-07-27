#pragma once
#include <CGUtils/math.hpp>
using namespace wzz::math;
using mat4 = wzz::math::mat4f_c;
using transform = mat4::right_transform;
class fps_camera_t
{
public:

	struct UpdateParams
	{
		bool front = false;
		bool left  = false;
		bool right = false;
		bool back  = false;

		bool up   = false;
		bool down = false;

		float cursor_rel_x = 0;
		float cursor_rel_y = 0;
	};

	fps_camera_t() noexcept{}

	void set_position(const vec3f &position) noexcept{
        this->pos = position;
    }

	void set_direction(float horiRad, float vertRad) noexcept{
        this->hori_rad = horiRad;
        this->vert_rad = vertRad;
    }

	void set_move_speed(float speed) noexcept{
        this->move_speed = speed;
    }

	void set_view_rotation_speed(float speed) noexcept{
        this->cursor_speed = speed;
    }

	void update(const UpdateParams &params) noexcept{
        vert_rad -= cursor_speed * params.cursor_rel_y;
        hori_rad += cursor_speed * params.cursor_rel_x;

        const float PI = wzz::math::PI_f;
        while(hori_rad < 0) hori_rad += 2 * PI;
        while(hori_rad >= 2 * PI) hori_rad -= 2 * PI;
        vert_rad = std::clamp(vert_rad,-PI * 0.5f + 0.01f,PI * 0.5f + 0.01f);

        const auto dir = get_xyz_direction();
        const auto front = vec3f(dir.x,0,dir.z).normalized();
        const auto right = cross(dir,{0,1,0}).normalized();

        const int front_step = params.front - params.back;
        const int right_step = params.right - params.left;

        pos += move_speed * ((float)front_step * front + (float)right_step * right);


        if(params.up)
            pos.y += move_speed;
        if(params.down)
            pos.y -= move_speed;
    }

	void set_perspective(float fovDeg, float nearZ, float farZ) noexcept{
            this->fov_deg = fovDeg;
            this->near_z = nearZ;
            this->far_z = farZ;
    }

	void set_w_over_h(float wOverH) noexcept{
        this->w_over_h = wOverH;
    }

	void recalculate_matrics() noexcept{
        const auto dir = get_xyz_direction();
        view = transform::look_at(pos,pos + dir,{0,1,0});
        proj = transform::perspective(wzz::math::deg2rad(fov_deg),w_over_h,near_z,far_z);
        view_proj = proj * view;
    }

	float get_near_z() const noexcept{
        return near_z;
    }

	float get_far_z() const noexcept{
        return far_z;
    }

	const vec3f &get_position() const noexcept{
        return pos;
    }

	/**
	 * @return {hori_radians,vert_radians}
	 */
	vec2f get_direction() const noexcept{
        return {hori_rad,vert_rad};
    }

	/**
	 * @return normalized direction in xyz-coord
	 */
	vec3f get_xyz_direction() const noexcept{
        float y = std::sin(vert_rad);
        float x = std::cos(vert_rad) * std::cos(hori_rad);
        float z = std::cos(vert_rad) * std::sin(hori_rad);
        return {x,y,z};
    }

	const mat4 &get_view() const noexcept{
        return view;
    }

	const mat4 &get_proj() const noexcept{
        return proj;
    }

	const mat4 &get_view_proj() const noexcept{
        return view_proj;
    }

private:


	vec3f pos;
	float vert_rad = 0.f;
	float hori_rad = 0.f;

	float fov_deg = 60.f;
	float near_z = 0.1f;
	float far_z = 100.f;

	float w_over_h = 1;

	float move_speed = 0.02f;
	float cursor_speed = 0.003f;

	mat4 view;
	mat4 proj;
	mat4 view_proj;
};
