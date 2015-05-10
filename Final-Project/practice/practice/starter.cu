// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010
// ://computer-graphics.se/hello-world-for-cuda.html

//Include GLEW
#include <GL/glew.h>

//Include GLFW
#include <GLFW/glfw3.h>

#include <stdio.h>

#define _USE_MATH_DEFINES
#include <math.h>

const int block_width = 1024;
const int block_height = 1;
const int MAX_BULLET_COUNT = 100000;
const int MAX_ENEMY_COUNT = 100;
const int MAX_SHOT_COUNT = 1000;
const int MAX_host_to_device_data_COUNT = 1000;
const int GAMEFIELD_SEMIWIDTH = 320;//6 * 32;
const int GAMEFIELD_SEMIHEIGHT = 240;//7 * 32;
const int PLAYER_TOLERANCE = -20;
const int BULLET_TOLERANCE = 50;
const int ENEMY_TOLERANCE = 100;
const int SHOT_TOLERANCE = 100;
const int PLAYER_FORM_TIME = 15;
const int PLAYER_ACTIVE_INVUL_TIME = 180;
const int PLAYER_FADE_TIME = 15;
const int PLAYER_DEAD_TIME = 30;
const int ENEMY_FORM_TIME = 0;
const int ENEMY_FADE_TIME = 15;
const int SHOT_FADE_TIME = 5;

enum entity_status { FORM, ACTIVE, FADE, DEAD };
enum bullet_type { BASIC };

//Define an error callback  
static void error_callback(int error, const char* description);

struct point;
struct polar_point;
struct host_data;

struct point{
	double x;
	double y;
	__host__ __device__	point();
	__host__ __device__	point(const point& p);
	__host__ __device__	point(double x, double y);
	__host__ __device__	point(const polar_point& p);
};

struct polar_point{
	double r;
	double t;
	__host__ __device__	polar_point();
	__host__ __device__	polar_point(const polar_point& p);
	__host__ __device__	polar_point(double r, double t);
	__host__ __device__	polar_point(const point& p);
};

point::point() : x(0), y(0){}
point::point(const point& p) : x(p.x), y(p.y){}
point::point(double x, double y) : x(x), y(y){}
point::point(const polar_point& p) {
	x = p.r*cos(p.t);
	y = p.r*sin(p.t);
}

__host__ __device__ point operator+(point & a, point & b){
	point return_value(a.x + b.x, a.y + b.y);
	return return_value;
}

__host__ __device__ point operator*(point & a, point & b){
	point return_value(a.x*b.x, a.y*b.y);
	return return_value;
}

__host__ __device__ point operator/(point & a, point & b){
	point return_value(a.x/b.x, a.y/b.y);
	return return_value;
}

__host__ __device__ point operator-(point & a, point & b){
	point return_value(a.x-b.x, a.y-b.y);
	return return_value;
}

polar_point::polar_point() : r(0), t(0){};
polar_point::polar_point(const polar_point& p) : r(p.r), t(p.t){}
polar_point::polar_point(double r, double t) : r(r), t(t){}
polar_point::polar_point(const point& p) {
	r = hypot(p.x, p.y);
	t = ((p.x != 0) || (p.y != 0)) ?
		atan2(p.y, p.x) :
		0;
}

struct shot{
	point pos;
	point vel;
	point semi_size;
	int damage;
	entity_status status;
	int age;
};

struct shot_container{
	shot shot_list[MAX_SHOT_COUNT];
	int shot_count;
};

typedef struct bullet_t{
	point pos;
	point vel;
	point acc;
	entity_status status;
	bullet_type type;
	int age;
	double theta;
	double w;
	__host__ __device__	bullet_t() : status(FORM), type(BASIC), age(0), w(0){}
	__host__ __device__	bullet_t(bullet_t& b) : pos(b.pos), vel(b.vel), acc(b.acc),
												status(b.status), type(b.type), age(b.age),
												theta(b.theta), w(b.w) {}
} bullet;

typedef struct pull_info_t{
    bool need;
	int delta;
} pull_info;

typedef struct bullet_slot_t{
	bullet load;
	pull_info pull;
} bullet_slot;

typedef struct bullet_draw_info_t{
	point pos;
	double theta;
	entity_status status;
	bullet_type type;
} bullet_draw_info;

typedef struct bullet_container_t{
	bullet_slot bullet_slots[MAX_BULLET_COUNT];
	int bullet_count;
	bool collision_with_player;
} bullet_container;

typedef struct device_to_host_data_t{
	bullet_draw_info draw_slots[MAX_BULLET_COUNT];
	int bullet_count;
	bool collision_with_player;
} device_to_host_data;

typedef struct host_to_device_data_t{
	bullet bullets[MAX_host_to_device_data_COUNT];
	int queue_count;
} host_to_device_data;

struct player{
	point pos;
	point vel;
	static const int radius = 3;
	entity_status status;
	int age;
	bool invul;
	bool is_hit;
	bool is_focus;
	bool is_shooting;
	__host__ __device__ player() : pos(0, -GAMEFIELD_SEMIHEIGHT*.8), vel(0, 0), status(FORM), age(0),
	invul(false){}
};

struct enemy{
	point pos;
	point vel;
	entity_status status;
	int hp;
	int age;
	double radius;
	void(*update) (enemy&, host_data&);
};

struct enemy_container{
	enemy enemy_list[MAX_ENEMY_COUNT];
	int enemy_count;
	__host__ __device__ enemy_container(): enemy_count(0) {}
};

typedef struct draw_data_t{
	bullet_draw_info* bullet_draw_infos;
	int* bullet_count;
	player* player_info;
} draw_data;

struct bullet_properties{
	point semi_size;
};

struct host_data{
	enemy_container enemies;
	host_to_device_data htd_data;
	player main_player;
	device_to_host_data dth_data;
	shot_container shots;
	int age;
	int deaths;
	int enemies_killed;
};

__device__ __host__ point rotate_point(const point& pt,const double theta){
	point return_value(cos(theta)*pt.x - sin(theta)*pt.y, 
					   cos(theta)*pt.y + sin(theta)*pt.x);
	return return_value;
}

__device__ __host__ point transpose_point(const point& pt){
	point return_value(pt.y,pt.x);
	return return_value;
}

__device__ __host__ bullet_properties get_bullet_properties(bullet& bullet){
	bullet_properties return_value;
	switch (bullet.type)
	{
	case BASIC:
		return_value.semi_size = point(2, 2);
		break;
	default:
		break;
	}
	return return_value;
}

__global__ void container_initialize_all_bullet(bullet_container* data,
	size_t bullet_count)
{
	data->bullet_count = bullet_count;
	int bullet_index = threadIdx.x;
	while (bullet_index < bullet_count){
		polar_point rt_vector = { 0, 0 };
		data->bullet_slots[bullet_index].load.pos = point(rt_vector);
		rt_vector.r = bullet_index*.0001;
		rt_vector.t = bullet_index;
		data->bullet_slots[bullet_index].load.vel = point(rt_vector);
		rt_vector.r = -.001;
		rt_vector.t = bullet_index;
		data->bullet_slots[bullet_index].load.acc = point(rt_vector);
		data->bullet_slots[bullet_index].load.theta = rt_vector.t;
		data->bullet_slots[bullet_index].load.w = -.001;
		data->bullet_slots[bullet_index].load.status = ACTIVE;
		bullet_index += blockDim.x;
	}
}

__global__ void container_add_new_bullets(bullet_container* data, host_to_device_data* new_bullets){
	int old_bullet_count = data->bullet_count;
	int add_bullet_count = new_bullets->queue_count;
	int bullet_index = threadIdx.x;
	while (bullet_index < add_bullet_count){
		data->bullet_slots[old_bullet_count + bullet_index].load = new_bullets->bullets[bullet_index];
		bullet_index += blockDim.x;
	}
	data->bullet_count = old_bullet_count + add_bullet_count;
}

__host__ __device__ bool in_bounds(point& pos, double tolerance){
	if (abs(pos.x) > GAMEFIELD_SEMIWIDTH + tolerance ||
		abs(pos.y) > GAMEFIELD_SEMIHEIGHT + tolerance){
		return false;
	}
	else {
		return true;
	}
}


__device__ void check_bounds_bullet(bullet* bullet){
	if (!in_bounds(bullet->pos, BULLET_TOLERANCE)){
		bullet->status = DEAD;
	}
}

__device__ void update_bullet(bullet* bullet){
	bullet->pos.x += bullet->vel.x;
	bullet->pos.y += bullet->vel.y;
	bullet->vel.x += bullet->acc.x;
	bullet->vel.y += bullet->acc.y;
	bullet->theta += bullet->w;
	bullet->age += 1;
	bullet->theta++;
	check_bounds_bullet(bullet);
}

__global__ void update_all_bullet(bullet* bullets,
	size_t bullet_count)
{
	int bullet_index = threadIdx.x;
	while (bullet_index < bullet_count){
		update_bullet(bullets + bullet_index);
		bullet_index += blockDim.x;
	}
}

__global__ void container_update_all_bullet(bullet_container* data)
{
	int bullet_index = threadIdx.x;
	int count = data->bullet_count;
	while (bullet_index < count){
		update_bullet(&(data->bullet_slots[bullet_index].load));
		bullet_index += blockDim.x;
	}
}

__global__ void mark_bullet_pull(bullet_container* container){
	int slot_range_width = 1 + (container->bullet_count - 1) / ((int)blockDim.x);
	int offset = 0;
	for (int i = 0; i < slot_range_width; ++i){
		int index = i + slot_range_width*threadIdx.x;
		if (container->bullet_slots[index].load.status == DEAD){
			++offset;
			container->bullet_slots[index].pull.need = false;
		} else {
			container->bullet_slots[index].pull.need = true;
		}
		container->bullet_slots[index].pull.delta = offset;
	}

	for (int k = 1; k <= blockDim.x; k = k << 1){
		__syncthreads();
		int delta = 0;
		if ((k & threadIdx.x) && ((threadIdx.x / k)*slot_range_width*k >= 1)){
			delta = container->bullet_slots[(threadIdx.x / k)*slot_range_width*k - 1].pull.delta;
		}
		for (int i = 0; i < slot_range_width; ++i){
			int index = i + slot_range_width*threadIdx.x;
			container->bullet_slots[index].pull.delta += delta;
		}
	}

}

__global__ void relocate_all_bullet(volatile bullet_container* container){
	
	int bullet_index = threadIdx.x;
	int count = container->bullet_count;
	int delta;
	while (bullet_index < count){
		bullet load;
		load = *((bullet*) &(container->bullet_slots[bullet_index].load));
		__syncthreads();
		delta = container->bullet_slots[bullet_index].pull.delta;
		if (container->bullet_slots[bullet_index].pull.need && (bullet_index - delta >= 0)){
			*((bullet*)&(container->bullet_slots[bullet_index-delta].load)) = load;
		}
		bullet_index += blockDim.x;
	}
	if (bullet_index == count - 1 + blockDim.x){
		container->bullet_count = count - delta;
	}

}

__device__ void extract_bullet_draw_info(bullet* bullet,
	bullet_draw_info* output){
	output->pos = bullet->pos;
	output->theta = bullet->theta;
	output->status = bullet->status;
	output->type = bullet->type;
}

__global__ void container_extract_all_bullet_draw_info(bullet_container* b_container,
													   device_to_host_data* d_container){
	int bullet_index = threadIdx.x;
	d_container->bullet_count = b_container->bullet_count;
	while (bullet_index < b_container->bullet_count){
		extract_bullet_draw_info(&(b_container->bullet_slots[bullet_index].load),
			&(d_container->draw_slots[bullet_index]));
		bullet_index += blockDim.x;
	}
	d_container->collision_with_player = b_container->collision_with_player;


}

__device__ bool collide_against_player(bullet& bullet, player& main_player){
	bullet_properties prop = get_bullet_properties(bullet);
	point dist_thresh = (prop.semi_size + point(main_player.radius, main_player.radius));
	point dist = rotate_point(bullet.pos - main_player.pos, -bullet.theta)* transpose_point(dist_thresh);
	if (dist.x*dist.x + dist.y*dist.y < dist_thresh.x*dist_thresh.x*dist_thresh.y*dist_thresh.y){
		bullet.status = DEAD;
		return true;
	}
	return false;

}

__global__ void container_collide_against_player(bullet_container* b_container,
	player main_player){
	int bullet_index = threadIdx.x;
	b_container->collision_with_player = false;
		bool collided = false;
		while (bullet_index < b_container->bullet_count){
			collided |= collide_against_player(b_container->bullet_slots[bullet_index].load, main_player);
			bullet_index += blockDim.x;
		}
		if (collided){
			b_container->collision_with_player = true;
		}
}

void pxlVertex2f(double x, double y){
	glVertex2f(x / GAMEFIELD_SEMIWIDTH, y / GAMEFIELD_SEMIHEIGHT);
}

void pxlVertexPos(const point& pt){
	glVertex2f(pt.x / GAMEFIELD_SEMIWIDTH, pt.y / GAMEFIELD_SEMIHEIGHT);
}



void gl_setup(GLFWwindow** window){

	//Set the error callback
	glfwSetErrorCallback(error_callback);

	//Initialize GLFW
	if (!glfwInit())
	{
		exit(EXIT_FAILURE);
	}

	//Set the GLFW window creation hints - these are optional
	//glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); //Request a specific OpenGL version
	//glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); //Request a specific OpenGL version
	glfwWindowHint(GLFW_SAMPLES, 4); //Request 4x antialiasing
	//glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	//Declare a window object


	//glEnable(GL_COLOR_MATERIAL);
	//Create a window and create its OpenGL context
	*window = glfwCreateWindow(2*GAMEFIELD_SEMIWIDTH, 2*GAMEFIELD_SEMIHEIGHT, "Test Window", NULL, NULL);

	//If the window couldn't be created
	if (!*window)
	{
		fprintf(stderr, "Failed to open GLFW window.\n");
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	//This function makes the context of the specified window current on the calling thread. 
	glfwMakeContextCurrent(*window);


	//Initialize GLEW
	GLenum err = glewInit();

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	//If GLEW hasn't initialized
	if (err != GLEW_OK)
	{
		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
	}

}



void queue_bullet(bullet& new_bullet, host_to_device_data& queue){
	if (queue.queue_count < MAX_host_to_device_data_COUNT){
		queue.bullets[queue.queue_count] = new_bullet;
		++(queue.queue_count);
	}
}

void queue_enemy(const enemy& new_enemy, host_data& game_data_h){
	game_data_h.enemies.enemy_list[game_data_h.enemies.enemy_count] = new_enemy;
	++game_data_h.enemies.enemy_count;
}

void queue_shot(const shot& new_shot, host_data& game_data_h){
	game_data_h.shots.shot_list[game_data_h.shots.shot_count] = new_shot;
	++game_data_h.shots.shot_count;
}

void draw_bullet(bullet_draw_info* info){
	//glClear(GL_COLOR_BUFFER_BIT);

	glBegin(GL_POLYGON);              // Each set of 4 vertices form a quad
	if (info->status == DEAD){
		glColor3f(1.0, 0.0, 0.0); // Red
	}
	else {
		glColor4f(1.0, 0.0, 1.0, 1.0);
	}
	for (int i = 0; i < 8; ++i){
		pxlVertexPos(info->pos + point(polar_point(5, i * 2 * M_PI / 8)));
	}
	glEnd();

	glBegin(GL_POLYGON);              // Each set of 4 vertices form a quad
	if (info->status == DEAD){
		glColor3f(1.0, 0.0, 0.0); // Red
	}
	else {
		glColor4f(1.0, 1.0, 1.0, .8);
	}
	for (int i = 0; i < 8; ++i){
		pxlVertexPos(info->pos + point(polar_point(4, i * 2 * M_PI / 8)));
	}
	glEnd();
}

void draw_player(player& main_player){
	glBegin(GL_QUADS);              // Each set of 4 vertices form a quad
	
		
		double ratio;
		switch (main_player.status){
		case FORM:
			ratio = ((double)main_player.age) / PLAYER_FORM_TIME;
			glColor4f(1.0, 0.5, 0.5, ratio);
			pxlVertex2f(main_player.pos.x + 25, main_player.pos.y + 25);
			pxlVertex2f(main_player.pos.x - 25, main_player.pos.y + 25);
			pxlVertex2f(main_player.pos.x - 25, main_player.pos.y - 25);
			pxlVertex2f(main_player.pos.x + 25, main_player.pos.y - 25);
			break;
		case ACTIVE:
			if (main_player.invul && (main_player.age & 4)){
				glColor4f(1.0, .6, .6, 1.0);
			}
			else {
				glColor4f(1.0, 0.5, 0.5, 1.0);
			}
			pxlVertex2f(main_player.pos.x + 25, main_player.pos.y + 25);
			pxlVertex2f(main_player.pos.x - 25, main_player.pos.y + 25);
			pxlVertex2f(main_player.pos.x - 25, main_player.pos.y - 25);
			pxlVertex2f(main_player.pos.x + 25, main_player.pos.y - 25);
			break;
		case FADE:
			ratio = ((double)main_player.age) / PLAYER_FADE_TIME;
			glColor4f(1.0, 0.5, 0.5, 1.0 - ratio);
			pxlVertex2f(main_player.pos.x + 25 * (1.0 + ratio), main_player.pos.y + 25 * (1.0 + ratio));
			pxlVertex2f(main_player.pos.x - 25 * (1.0 + ratio), main_player.pos.y + 25 * (1.0 + ratio));
			pxlVertex2f(main_player.pos.x - 25 * (1.0 + ratio), main_player.pos.y - 25 * (1.0 + ratio));
			pxlVertex2f(main_player.pos.x + 25 * (1.0 + ratio), main_player.pos.y - 25 * (1.0 + ratio));
			break;
		case DEAD:
			glColor4f(0.0, 0.0, 0.0, .0);
			break;
		default:
			glColor3f(0.0, 0.0, 0.0);
			break;
		}
	
	
	glEnd();
}

void draw_player_hitbox(player& main_player){
	glBegin(GL_QUADS);
	glColor3f(1.0, 1.0, 1.0);
	pxlVertex2f(main_player.pos.x + 3, main_player.pos.y + 3);
	pxlVertex2f(main_player.pos.x - 3, main_player.pos.y + 3);
	pxlVertex2f(main_player.pos.x - 3, main_player.pos.y - 3);
	pxlVertex2f(main_player.pos.x + 3, main_player.pos.y - 3);
	glEnd();
}

void draw_enemy(enemy& enemy){
	glBegin(GL_QUADS);
	double mult = 1;
	glColor3f(1.0, 0.5, 0.0);
	if (enemy.status == FADE){
		double ratio = (1.0*enemy.age) / (ENEMY_FADE_TIME);
		mult = 1 + ratio;
		glColor4f(1.0, 0.5, 0.0, 1-ratio);
	}
	pxlVertex2f(enemy.pos.x + mult*enemy.radius, enemy.pos.y + mult*enemy.radius);
	pxlVertex2f(enemy.pos.x - mult*enemy.radius, enemy.pos.y + mult*enemy.radius);
	pxlVertex2f(enemy.pos.x - mult*enemy.radius, enemy.pos.y - mult*enemy.radius);
	pxlVertex2f(enemy.pos.x + mult*enemy.radius, enemy.pos.y - mult*enemy.radius);
	glEnd();
}

void draw_shot(shot& shot){
	glBegin(GL_QUADS);
	double t = polar_point(shot.vel).t - M_PI_2;
	point c1 = rotate_point(point(1, 1)*shot.semi_size, t);
	point c2 = rotate_point(point(1, -1)*shot.semi_size, t);
	point c3 = rotate_point(point(-1, -1)* shot.semi_size, t);
	point c4 = rotate_point(point(-1, 1)*shot.semi_size, t);
	if (shot.status == ACTIVE)
	{
		glColor4f(0.5, 0.5, 1.0, .5);
		pxlVertexPos(shot.pos + c1);
		pxlVertexPos(shot.pos + c2);
		pxlVertexPos(shot.pos + c3);
		pxlVertexPos(shot.pos + c4);
	}
	if (shot.status == FADE){
		double ratio = (1.0*shot.age) / SHOT_FADE_TIME;
		point ratio_pt(1.0 + ratio, 1.0 + ratio);
		glColor4f(0.5, 0.5, 1.0, .5*(1-ratio));
		pxlVertexPos(shot.pos + c1*ratio_pt);
		pxlVertexPos(shot.pos + c2*ratio_pt);
		pxlVertexPos(shot.pos + c3*ratio_pt);
		pxlVertexPos(shot.pos + c4*ratio_pt);
	}
	
	
	
	glEnd();
}

void draw_screen(host_data* game_data){
	glClear(GL_COLOR_BUFFER_BIT);
	draw_player(game_data->main_player);
	for (int i = 0; i < game_data->enemies.enemy_count; ++i){
		draw_enemy((game_data->enemies.enemy_list[i]));
	}
	for (int i = 0; i < game_data->shots.shot_count; ++i){
		draw_shot((game_data->shots.shot_list[i]));
	}
	for (int i = game_data->dth_data.bullet_count - 1; i >= 0; --i){
		draw_bullet(&(game_data->dth_data.draw_slots[i]));
	}
	draw_player_hitbox(game_data->main_player);
}


__device__ __host__ point set_mag_point(point& pt, double mag){
	polar_point polar_pt(pt);
	polar_pt.r = mag;
	point return_value(polar_pt);
	return return_value;
}



void set_player_velocity(player& main_player, GLFWwindow *window){
	if (glfwGetKey(window, GLFW_KEY_UP)){
		main_player.vel.y = 1;
	}
	else if (glfwGetKey(window, GLFW_KEY_DOWN)){
		main_player.vel.y = -1;
	}
	else {
		main_player.vel.y = 0;
	}

	if (glfwGetKey(window, GLFW_KEY_RIGHT)){
		main_player.vel.x = 1;
	}
	else if (glfwGetKey(window, GLFW_KEY_LEFT)){
		main_player.vel.x = -1;
	}
	else {
		main_player.vel.x = 0;
	}

	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)){
		main_player.is_focus = true;
		if (main_player.vel.x != 0 || main_player.vel.y != 0){
			main_player.vel = set_mag_point(main_player.vel, 1.0);
		}
	}
	else {
		main_player.is_focus = false;
		if (main_player.vel.x != 0 || main_player.vel.y != 0){
			main_player.vel = set_mag_point(main_player.vel, 3.0);
		}
	}

	if (glfwGetKey(window, GLFW_KEY_Z)){
		main_player.is_shooting = true;
	}
	else {
		main_player.is_shooting = false;
	}
}

void move_player(player& main_player){
	main_player.pos.x += main_player.vel.x;
	main_player.pos.y += main_player.vel.y;
	if (abs(main_player.pos.x) > GAMEFIELD_SEMIWIDTH + PLAYER_TOLERANCE){
		main_player.pos.x = (GAMEFIELD_SEMIWIDTH + PLAYER_TOLERANCE) *
			((main_player.pos.x > 0) - (main_player.pos.x < 0));
	}
	if (abs(main_player.pos.y) > GAMEFIELD_SEMIHEIGHT + PLAYER_TOLERANCE){
		main_player.pos.y = (GAMEFIELD_SEMIHEIGHT + PLAYER_TOLERANCE) * 
			((main_player.pos.y > 0) - (main_player.pos.y < 0));
	}
	
}

void update_player(player& main_player, host_data& game_data_h){
	main_player.is_hit |= game_data_h.dth_data.collision_with_player;
	switch (main_player.status){
	case FORM:
		main_player.invul = true;
		if (main_player.age > PLAYER_FORM_TIME){
			main_player.status = ACTIVE;
			main_player.age = 0;
		}
		break;
	case ACTIVE:
		if (main_player.age == PLAYER_ACTIVE_INVUL_TIME){
			main_player.invul = false;
		}
		move_player(main_player);
		if (main_player.is_hit && !main_player.invul){
			main_player.status = FADE;
			main_player.age = 0;
		}
		if (main_player.age % 6 == 0 && main_player.is_shooting){
			shot new_shot;
			new_shot.damage = 8;
			new_shot.semi_size = point(3, 12);
			new_shot.status = FORM;
			new_shot.age = 0;

			double spread = main_player.is_focus ? 1.0 : 3.0;

			new_shot.vel = point(0, 10);

			new_shot.pos = main_player.pos + point(-15, -10);
			queue_shot(new_shot, game_data_h);

			new_shot.pos = main_player.pos + point(15, -10);
			queue_shot(new_shot, game_data_h);

			new_shot.vel = polar_point(8, M_PI_2 + .02*spread);
			new_shot.pos = main_player.pos + point(-4 * spread, 5);
			queue_shot(new_shot, game_data_h);

			new_shot.vel = polar_point(8, M_PI_2 - .02*spread);
			new_shot.pos = main_player.pos + point(4 * spread, 5);
			queue_shot(new_shot, game_data_h);

			new_shot.vel = polar_point(8, M_PI_2 - .06*spread);
			new_shot.pos = main_player.pos + point(12 * spread, 5);
			queue_shot(new_shot, game_data_h);

			new_shot.vel = polar_point(8, M_PI_2 + .06*spread);
			new_shot.pos = main_player.pos + point(-12 * spread, 5);
			queue_shot(new_shot, game_data_h);


		}
		break;
	case FADE:
		if (main_player.age > PLAYER_FADE_TIME){
			main_player.status = DEAD;
			main_player.age = 0;
			game_data_h.deaths++;
		}
		break;
	case DEAD:
		if (main_player.age > PLAYER_DEAD_TIME){
			main_player.status = FORM;
			main_player.age = 0;
		}
		break;
	default:
		break;
	}
	main_player.is_hit = false;
	
	++main_player.age;
}

void generic_enemy_update(enemy& self, host_data& game_data_h){
	point diff;
	switch (self.status){
	case FORM:
		if (self.age > ENEMY_FORM_TIME){
			self.status = ACTIVE;
			self.age = 0;
		}
		break;
	case ACTIVE:
		self.pos = self.pos + self.vel;
		diff = self.pos - game_data_h.main_player.pos;
		if (abs(diff.x) < self.radius + game_data_h.main_player.radius &&
			abs(diff.y) < self.radius + game_data_h.main_player.radius){
			game_data_h.main_player.is_hit = true;
		}
		if (!in_bounds(self.pos, ENEMY_TOLERANCE) || self.hp <= 0){
			self.status = FADE;
			self.age = 0;
			if (self.hp <= 0){
				game_data_h.enemies_killed++;
			}
		}
		break;
	case FADE:
		if (self.age > ENEMY_FADE_TIME){
			self.status = DEAD;
			self.age = 0;
		}
		break;
	case DEAD:
		break;
	default:
		break;
	}
	++self.age;
}

void update_function_1(enemy& self, host_data& game_data_h){
	generic_enemy_update(self, game_data_h);
	if (self.age % 60 == 30){
		bullet sample;
		sample.pos = self.pos;
		polar_point diff = game_data_h.main_player.pos - self.pos;
		for (int dir = -1; dir <= 1; dir += 2){
			for (int mag = 0; mag < 4; ++mag){
				polar_point new_vel((.5+.3*mag), diff.t + dir*.1);
				sample.vel = new_vel;
				queue_bullet(sample, game_data_h.htd_data);
			}
		}
	}
}

void update_function_2(enemy& self, host_data& game_data_h){
	generic_enemy_update(self, game_data_h);
	if (self.age % 120 == 0 && self.pos.y > GAMEFIELD_SEMIHEIGHT*.3){
		bullet sample;
		polar_point diff = game_data_h.main_player.pos - self.pos;
		for (int dir = 0; dir < 32; ++dir){
			double t = diff.t + dir * 2 * M_PI / 32;
			sample.pos = self.pos + point(polar_point(50,t));
			for (int j = -1; j <= 1; j += 2){
				for (int mag = 5; mag <= 11; mag += 1){
					polar_point new_vel(mag*.1*j, t + M_PI_2);
					sample.vel = new_vel;
					queue_bullet(sample, game_data_h.htd_data);
				}
			}
		}
	}
}

void update_function_3(enemy& self, host_data& game_data_h){
	generic_enemy_update(self, game_data_h);
	if (self.age % 60 == 0 && self.pos.y > GAMEFIELD_SEMIHEIGHT*.3){
		bullet sample;
		polar_point diff = game_data_h.main_player.pos - self.pos;
		for (int dir = -3; dir <= 3; ++dir){
			double t = diff.t + dir * .2;
			sample.pos = self.pos;
				for (int mag = 3; mag <= 5; mag += 1){
					polar_point new_vel(mag*.2, t);
					sample.vel = new_vel;
					queue_bullet(sample, game_data_h.htd_data);
				}
		}
	}
}

void update_function_4(enemy& self, host_data& game_data_h){
	generic_enemy_update(self, game_data_h);
	if (self.age % 10 == 0 && self.pos.y > GAMEFIELD_SEMIHEIGHT*-.1){
		bullet sample;
		for (int dir = 0; dir < 8; ++dir){
			double t = dir * 2 * M_PI / 8 + self.age;
			sample.pos = self.pos + point(polar_point(30, t + M_PI_2));
			polar_point new_acc(.001, t);
			sample.acc = new_acc;
			queue_bullet(sample, game_data_h.htd_data);
		}
	}
}

void update_function_5(enemy& self, host_data& game_data_h){
	generic_enemy_update(self, game_data_h);
	int arms = 7;
	if (self.age % 2 == 0 && self.age > 300){
		bullet sample;
		self.vel = point(0, 0);
		for (int dir = 0; dir < arms; ++dir){
			double t = dir * 2 * M_PI / arms + 0.0002*self.age*self.age;
			sample.pos = self.pos;
			polar_point new_vel(1.0, t);
			sample.vel = new_vel;
			polar_point new_acc(-.005, t);
			sample.acc = new_acc;
			queue_bullet(sample, game_data_h.htd_data);
		}
	}
}

enemy enemy1(point pos, point vel){
	enemy return_value;
	return_value.pos = pos;
	return_value.vel = vel;
	return_value.status = FORM;
	return_value.radius = 20;
	return_value.update = update_function_2;
	return_value.hp = 2000;
	return_value.age = 0;
	return return_value;
}

enemy set_enemy(point pos, point vel, int hp, void(*update_function)(enemy&, host_data&)){
	enemy return_value;
	return_value.pos = pos;
	return_value.vel = vel;
	return_value.status = FORM;
	return_value.radius = 20;
	return_value.update = update_function;
	return_value.hp = hp;
	return_value.age = 0;
	return return_value;
}

void update_enemies(host_data& game_data_h){
	for (int i = 0; i < game_data_h.enemies.enemy_count; ++i){
		game_data_h.enemies.enemy_list[i].update(
			game_data_h.enemies.enemy_list[i], game_data_h);
	}
	int j = 0;
	for (int i = 0; i < game_data_h.enemies.enemy_count; ++i){

		if (game_data_h.enemies.enemy_list[i].status != DEAD){
			game_data_h.enemies.enemy_list[j] = game_data_h.enemies.enemy_list[i];
			++j;
		}
	}
	game_data_h.enemies.enemy_count = j;

}

void update_shot(shot& shot, host_data& game_data_h){
	point diff;
	polar_point polar_vel;
	switch (shot.status){
	case FORM:
			shot.status = ACTIVE;
			shot.age = 0;
		break;
	case ACTIVE:
		shot.pos = shot.pos + shot.vel;
		polar_vel = shot.vel;
		for (int i = 0; i < game_data_h.enemies.enemy_count; ++i){
			diff = rotate_point(game_data_h.enemies.enemy_list[i].pos - shot.pos, -polar_vel.t);
			if (abs(diff.x) < shot.semi_size.x + game_data_h.enemies.enemy_list[i].radius &&
				abs(diff.y) < shot.semi_size.y + game_data_h.enemies.enemy_list[i].radius){
				game_data_h.enemies.enemy_list[i].hp -= shot.damage;
				shot.status = FADE;
				shot.age = 0;
			}

		}
		if (!in_bounds(shot.pos, SHOT_TOLERANCE)){
			shot.status = FADE;
			shot.age = 0;
		}
		break;
	case FADE:
		shot.pos = shot.pos + shot.vel;
		if (shot.age > SHOT_FADE_TIME){
			shot.status = DEAD;
			shot.age = 0;
		}
		break;
	case DEAD:
		break;
	default:
		break;
	}
	++shot.age;
}

void update_shots(host_data& game_data_h){
	for (int i = 0; i < game_data_h.shots.shot_count; ++i){
		update_shot(
			game_data_h.shots.shot_list[i], game_data_h);
	}
	int j = 0;
	for (int i = 0; i < game_data_h.shots.shot_count; ++i){

		if (game_data_h.shots.shot_list[i].status != DEAD){
			game_data_h.shots.shot_list[j] = game_data_h.shots.shot_list[i];
			++j;
		}
	}
	game_data_h.shots.shot_count = j;

}

void game_script(host_data& game_data_h){
	
	
	if (game_data_h.age == 3000){

		queue_enemy(enemy1(point(GAMEFIELD_SEMIWIDTH*-.3, GAMEFIELD_SEMIHEIGHT*1.1), 
			polar_point(.3, 3 * M_PI_2)), game_data_h);
	}

	if (game_data_h.age == 2200){
		queue_enemy(enemy1(point(GAMEFIELD_SEMIWIDTH*.3, GAMEFIELD_SEMIHEIGHT*1.1),
			polar_point(.3, 3 * M_PI_2)), game_data_h);
	}

	if (game_data_h.age == 1120 || game_data_h.age == 1120 + 60 || game_data_h.age == 1120 + 120){
		queue_enemy(set_enemy(point(GAMEFIELD_SEMIWIDTH*-.7, GAMEFIELD_SEMIHEIGHT*1.1),
			polar_point(1.5, 7.5 * M_PI_4), 300, update_function_4), game_data_h);
	}

	if (game_data_h.age == 1480 || game_data_h.age == 1480 + 60 || game_data_h.age == 1480 + 120){
		queue_enemy(set_enemy(point(GAMEFIELD_SEMIWIDTH*.7, GAMEFIELD_SEMIHEIGHT*1.1),
			polar_point(1.5, 4.5 * M_PI_4), 300, update_function_4), game_data_h);
	}

	if (game_data_h.age == 120 || game_data_h.age == 120 + 60 || game_data_h.age == 120 + 120){
		queue_enemy(set_enemy(point(GAMEFIELD_SEMIWIDTH*-.7, GAMEFIELD_SEMIHEIGHT*1.1),
			polar_point(1.5, 6.5 * M_PI_4), 300, update_function_3), game_data_h);
	}

	if (game_data_h.age == 480 || game_data_h.age == 480 + 60 || game_data_h.age == 480 + 120){
		queue_enemy(set_enemy(point(GAMEFIELD_SEMIWIDTH*.7, GAMEFIELD_SEMIHEIGHT*1.1),
			polar_point(1.5, 5.5 * M_PI_4), 300, update_function_3), game_data_h);
	}

	if (game_data_h.age >= 4000 && game_data_h.age <= 4500 && game_data_h.age%50==0){
		queue_enemy(set_enemy(point(GAMEFIELD_SEMIWIDTH*.1, GAMEFIELD_SEMIHEIGHT*1.1),
			polar_point(2.5, 7 * M_PI_4), 100, update_function_1), game_data_h);
	}

	if (game_data_h.age == 5000){
		for (int i = 0; i < 5; ++i){
			queue_enemy(set_enemy(point(GAMEFIELD_SEMIWIDTH*-.7, GAMEFIELD_SEMIHEIGHT*1.1),
				polar_point(1.0, (6 + .5*i) * M_PI_4), 200, update_function_4), game_data_h);
		}
	}

	if (game_data_h.age == 6000){
		for (int i = 0; i < 5; ++i){
			queue_enemy(set_enemy(point(GAMEFIELD_SEMIWIDTH*+.7, GAMEFIELD_SEMIHEIGHT*1.1),
				polar_point(1.0, (6 - .5*i) * M_PI_4), 200, update_function_4), game_data_h);
		}
	}

	if (game_data_h.age == 7000){
			queue_enemy(set_enemy(point(GAMEFIELD_SEMIWIDTH*+.3, GAMEFIELD_SEMIHEIGHT*1.1),
				polar_point(.3, (6) * M_PI_4), 700, update_function_2), game_data_h);
			queue_enemy(set_enemy(point(GAMEFIELD_SEMIWIDTH*-.7, GAMEFIELD_SEMIHEIGHT*1.1),
				polar_point(1.0, (6 + .5*1) * M_PI_4), 100, update_function_4), game_data_h);
	}

	if (game_data_h.age == 8000){
		queue_enemy(set_enemy(point(GAMEFIELD_SEMIWIDTH*-.3, GAMEFIELD_SEMIHEIGHT*1.1),
			polar_point(.3, (6) * M_PI_4), 700, update_function_2), game_data_h);
		queue_enemy(set_enemy(point(GAMEFIELD_SEMIWIDTH*+.7, GAMEFIELD_SEMIHEIGHT*1.1),
			polar_point(1.0, (6 - .5 * 1) * M_PI_4), 100, update_function_4), game_data_h);
	}

	if (game_data_h.age == 9000){
		queue_enemy(set_enemy(point(GAMEFIELD_SEMIWIDTH*-.6, GAMEFIELD_SEMIHEIGHT*1.1),
			polar_point(.3, (6) * M_PI_4), 700, update_function_2), game_data_h);
		queue_enemy(set_enemy(point(GAMEFIELD_SEMIWIDTH*+.6, GAMEFIELD_SEMIHEIGHT*1.1),
			polar_point(.3, (6) * M_PI_4), 700, update_function_2), game_data_h);
	}

	if (game_data_h.age == 9060){
		queue_enemy(set_enemy(point(GAMEFIELD_SEMIWIDTH * 0, GAMEFIELD_SEMIHEIGHT*1.1),
			polar_point(.3, (6) * M_PI_4), 700, update_function_2), game_data_h);
	}




	if (game_data_h.age == 10000){
		queue_enemy(set_enemy(point(GAMEFIELD_SEMIWIDTH * 0, GAMEFIELD_SEMIHEIGHT*1.1),
			polar_point(0.5, (6) * M_PI_4), 10000, update_function_5), game_data_h);
	}

	++game_data_h.age;
}



int main()
{
	bullet_container* data_d;
	device_to_host_data* draw_d;
	//device_to_host_data* draw_h;
	host_to_device_data* new_host_to_device_data_d;
	//host_to_device_data* new_host_to_device_data_h;
	host_data* game_data_h;


	GLFWwindow* window;
	gl_setup(&window);
	const int bullets_count = 0;
	const int bullets_size = MAX_BULLET_COUNT*sizeof(bullet);
	const int bullet_draw_infos_size = MAX_BULLET_COUNT*sizeof(bullet_draw_info);

	dim3 dimBlock(block_width, block_height);
	dim3 dimGrid(1, 1);


	cudaMalloc((void**)&data_d, sizeof(bullet_container));
	cudaMalloc((void**)&draw_d, sizeof(bullet_container));
	cudaMalloc((void**)&new_host_to_device_data_d, sizeof(host_to_device_data));
	/*cudaMallocHost((void**)&draw_h, sizeof(device_to_host_data));
	cudaMallocHost((void**)&new_host_to_device_data_h, sizeof(host_to_device_data));*/
	cudaMallocHost((void**)&game_data_h, sizeof(host_data));
	/*draw_data_pointers.bullet_count = &(draw_h->bullet_count);
	draw_data_pointers.bullet_draw_infos = (draw_h->draw_slots);
	draw_data_pointers.player_info = &(main_player);*/
	game_data_h->age = 0;
	game_data_h->main_player = player();

	cudaStream_t stream1, stream2, stream3, stream4;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);
	cudaStreamCreate(&stream4);

	container_initialize_all_bullet <<<dimGrid, dimBlock, 0, stream4 >>>(
		data_d,
		bullets_count);

	cudaStreamSynchronize(stream4);

	//Set a background color
	glClearColor(0.0f, 0.0f, .5f, 0.0f);

	double time = glfwGetTime();
	const double FRAME_PERIOD = 1.0l / 60.0l;
	
	game_data_h->htd_data.queue_count = 0;
	//Main Loop
	do
	{
		if (glfwGetTime() - time >= FRAME_PERIOD){
			printf("Frame Rate: %f\n Bullets: %d\n Deaths: %d\n Enemies Killed: %d\n",
				1.0 / (glfwGetTime() - time),
				game_data_h->dth_data.bullet_count,
				game_data_h->deaths,
				game_data_h->enemies_killed
				);
			time = glfwGetTime();

			//test_queue_bullet(game_data_h->htd_data);

			// move bullets to queue
			if (cudaSuccess != cudaMemcpyAsync(new_host_to_device_data_d,
				&(game_data_h->htd_data),
				sizeof(host_to_device_data), cudaMemcpyHostToDevice, stream4)){
				printf("failure memcpy htd\n");
				return 1;
			}
			update_shots(*game_data_h);
			update_enemies(*game_data_h);
			game_script(*game_data_h);
			
			// reset queue
			

			cudaDeviceSynchronize();
			game_data_h->htd_data.queue_count = 0;
			container_extract_all_bullet_draw_info << <dimGrid, dimBlock, 0, stream2 >> >(
				data_d, draw_d);

			glClear(GL_COLOR_BUFFER_BIT);
			draw_screen(game_data_h);


			cudaDeviceSynchronize();

			container_update_all_bullet << <dimGrid, dimBlock, 0, stream1 >> >(data_d);
			container_collide_against_player << <dimGrid, dimBlock, 0, stream1 >> >(data_d, game_data_h->main_player);
			container_add_new_bullets << <dimGrid, dimBlock, 0, stream1 >> >(data_d, new_host_to_device_data_d);
			set_player_velocity(game_data_h->main_player, window);
			update_player(game_data_h->main_player, *game_data_h);
			cudaDeviceSynchronize();
			mark_bullet_pull << <dimGrid, dimBlock, 0, stream1 >> >(data_d);
			relocate_all_bullet << <dimGrid, dimBlock, 0, stream1 >> >(data_d);
			if (cudaSuccess != cudaMemcpyAsync(&(game_data_h->dth_data), draw_d,
				sizeof(device_to_host_data), cudaMemcpyDeviceToHost, stream3)){
				printf("failure memcpy dth\n");
				return 1;
			}

			
			
			
			//Swap buffers
			glfwSwapBuffers(window);
			//Get and organize events, like keyboard and mouse input, window resizing, etc...
			glfwPollEvents();
		}
	} //Check if the ESC key had been pressed or if the window had been closed
	while (!glfwWindowShouldClose(window));
	cudaFree(data_d);
	cudaFree(draw_d);
	cudaFree(new_host_to_device_data_d);
	cudaFreeHost(game_data_h);

	//Close OpenGL window and terminate GLFW
	glfwDestroyWindow(window);
	//Finalize and clean up GLFW
	glfwTerminate();




	exit(EXIT_SUCCESS);

}

//Define an error callback  
static void error_callback(int error, const char* description)
{
	fputs(description, stderr);
	_fgetchar();
}
