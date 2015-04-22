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
#include <math.h>

//Define an error callback  
static void error_callback(int error, const char* description)
{
	fputs(description, stderr);
	_fgetchar();
}

//Define the key input callback  
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);
}


const int block_width = 1024;
const int block_height = 1;
const int MAX_BULLET_COUNT = 1000000;
const int GAMEFIELD_SEMIWIDTH = 6*32;
const int GAMEFIELD_SEMIHEIGHT = 7*32;

typedef enum { FORM, ACTIVE, FADE, DEAD } bullet_status;
typedef enum { BASIC } bullet_type;


typedef struct point_t{
	double x;
	double y;
} point;

typedef struct bullet_t{
	point pos;
	point vel;
	point acc;
	bullet_status status;
	bullet_type type;
	int age;
	double theta;
	double w;
} bullet;

typedef struct pull_info_t{
    bool need;
	int delta;
} pull_info;

typedef struct bullet_slot_t{
	bullet load;
	pull_info pull;
} bullet_slot;

typedef struct draw_info_t{
	point pos;
	double theta;
	bullet_status status;
	bullet_type type;
} draw_info;

typedef struct data_container_t{
	bullet_slot bullet_slots[MAX_BULLET_COUNT];
	draw_info draw_slots[MAX_BULLET_COUNT];
	int bullet_count;
} data_container;

__device__ __host__ point xy_to_rt(point* xy_point){
	double r = hypot(xy_point->x, xy_point->y);
	double t = ((xy_point->x != 0) || (xy_point->y != 0)) ?
		atan2(xy_point->y, xy_point->x) :
		0;
	point return_value = { r, t };
	return return_value;
}

__device__ __host__ point rt_to_xy(point* rt_point){
	double r = rt_point->x;
	double t = rt_point->y;
	point return_value = { r*cos(t), r*sin(t) };
	return return_value;
}

__device__ __host__ point rt_point(double r, double t){
	point return_value = { r*cos(t), r*sin(t) };
	return return_value;
}

__device__ __host__ point xy_point(double x, double y){
	point return_value = {x, y};
	return return_value;
}

__global__ void initialize_all_bullet(bullet* bullets,
	size_t bullet_count)
{
	int bullet_index = threadIdx.x;
	while (bullet_index < bullet_count){
		point rt_vector = {0, 0};
		bullets[bullet_index].pos = rt_to_xy(&rt_vector);
		rt_vector.x = bullet_index*.01;
		rt_vector.y = bullet_index;
		bullets[bullet_index].vel = rt_to_xy(&rt_vector);
		rt_vector.x = -.001;
		rt_vector.y = bullet_index;
		bullets[bullet_index].acc = rt_to_xy(&rt_vector);
		bullets[bullet_index].theta = rt_vector.y;
		bullets[bullet_index].w = -.001;

		bullet_index += blockDim.x;
	}
}

__device__ void check_bounds_bullet(bullet* bullet){
	if (abs(bullet->pos.x) > GAMEFIELD_SEMIWIDTH ||
		abs(bullet->pos.y) > GAMEFIELD_SEMIHEIGHT){
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

__global__ void mark_bullet_pull(data_container* container){
	int slot_range_width = 1 + (container->bullet_count - 1) / blockDim.x;
	bool copy = 0;
	container->bullet_count;
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
		if (k & threadIdx.x){
			delta = container->bullet_slots[(threadIdx % k)*slot_range_width - 1].pull.delta;
		}
		for (int i = 0; i < slot_range_width; ++i){
			int index = i + slot_range_width*threadIdx.x;
			container->bullet_slots[index].pull.delta += delta;
		}
	}

}

__device__ void extract_bullet_draw_info(bullet* bullet,
	draw_info* output){
	output->pos = bullet->pos;
	output->theta = bullet->theta;
	output->status = bullet->status;
	output->type = bullet->type;
}

__global__ void extract_all_bullet_draw_info(bullet* bullets,
	draw_info* output,
	size_t bullet_count){
	int bullet_index = threadIdx.x;
	while (bullet_index < bullet_count){
		extract_bullet_draw_info(bullets + bullet_index,
			output + bullet_index);
		bullet_index += blockDim.x;
	}
}

__global__ void container_extract_all_bullet_draw_info(data_container* container){
	int bullet_index = threadIdx.x;
	while (bullet_index < container->bullet_count){
		extract_bullet_draw_info(&(container->bullet_slots[bullet_index].load),
			&(container->draw_slots[bullet_index]));
		bullet_index += blockDim.x;
	}
}

void pxlVertex2f(double x, double y){
	glVertex2f(x / GAMEFIELD_SEMIWIDTH, y / GAMEFIELD_SEMIHEIGHT);
}

void pxlVertexPos(point* pt){
	glVertex2f(pt->x / GAMEFIELD_SEMIWIDTH, pt->y / GAMEFIELD_SEMIHEIGHT);
}



void draw_object(draw_info* info){

	//glClear(GL_COLOR_BUFFER_BIT);
	glBegin(GL_TRIANGLES);              // Each set of 4 vertices form a quad
	if (info->status == DEAD){
		glColor3f(1.0, 0.0, 0.0); // Red
	}
	else {
		glColor3f(0.0, 1.0, 0.0);
	}
		
		pxlVertex2f(info->pos.x + 4, info->pos.y + 4);
		pxlVertex2f(info->pos.x - 4, info->pos.y + 4);
		pxlVertex2f(info->pos.x, info->pos.y - 4);
	glEnd();
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
	//glfwWindowHint(GLFW_SAMPLES, 4); //Request 4x antialiasing
	//glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	//Declare a window object


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

	//Sets the key callback
	glfwSetKeyCallback(*window, key_callback);

	//Initialize GLEW
	GLenum err = glewInit();

	//If GLEW hasn't initialized
	if (err != GLEW_OK)
	{
		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
	}

}



int main()
{

	bullet* bullets_d;
	draw_info* draw_infos_d;
	draw_info* draw_infos_h;


	GLFWwindow* window;
	gl_setup(&window);
	const int bullets_count = 50000;
	const int bullets_size = MAX_BULLET_COUNT*sizeof(bullet);
	const int draw_infos_size = MAX_BULLET_COUNT*sizeof(draw_info);

	dim3 dimBlock(block_width, block_height);
	dim3 dimGrid(1, 1);


	cudaMalloc((void**)&bullets_d, bullets_size);
	cudaMalloc((void**)&draw_infos_d, draw_infos_size);
	cudaMallocHost((void**)&draw_infos_h, draw_infos_size);
	draw_infos_h = (draw_info*)malloc(draw_infos_size);




	cudaStream_t stream1, stream2, stream3, stream4;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);
	cudaStreamCreate(&stream4);

	initialize_all_bullet <<<dimGrid, dimBlock, 0, stream4 >>>(
		bullets_d,
		bullets_count);

	cudaStreamSynchronize(stream4);

	//Set a background color
	glClearColor(0.0f, 0.0f, 1.0f, 0.0f);

	double time = glfwGetTime();
	const double FRAME_PERIOD = 1.0l / 60.0l;
	//Main Loop
	do
	{
		if (glfwGetTime() - time >= FRAME_PERIOD){
			printf("%f\n", 1.0 / (glfwGetTime() - time));
			time = glfwGetTime();
			extract_all_bullet_draw_info << <dimGrid, dimBlock, 0, stream2 >> >(
				bullets_d,
				draw_infos_d,
				bullets_count);
			glClear(GL_COLOR_BUFFER_BIT);
			for (int i = 0; i < bullets_count; ++i){
				draw_object(&(draw_infos_h[i]));
			}
			cudaDeviceSynchronize();

			update_all_bullet << <dimGrid, dimBlock, 0, stream1 >> >(
				bullets_d,
				bullets_count);

			if (cudaSuccess != cudaMemcpyAsync(draw_infos_h, draw_infos_d,
				draw_infos_size, cudaMemcpyDeviceToHost, stream3)){
				printf("failure \n");
			}

			cudaDeviceSynchronize();
			
			
			//Swap buffers
			glfwSwapBuffers(window);
			//Get and organize events, like keyboard and mouse input, window resizing, etc...
			glfwPollEvents();
		}
	} //Check if the ESC key had been pressed or if the window had been closed
	while (!glfwWindowShouldClose(window));
	cudaFree(bullets_d);
	cudaFree(draw_infos_d);
	cudaFreeHost(draw_infos_h);

	//Close OpenGL window and terminate GLFW
	glfwDestroyWindow(window);
	//Finalize and clean up GLFW
	glfwTerminate();




	exit(EXIT_SUCCESS);


	/*
Bullet #900 x: 900.040100 y: 899.999900
Bullet #900 x: 900.040200 y: 899.999800
Bullet #900 x: 900.040300 y: 899.999700
Bullet #900 x: 900.040400 y: 899.999600
Bullet #900 x: 900.040500 y: 899.999500
Bullet #900 x: 900.040600 y: 899.999400
Bullet #900 x: 900.040600 y: 899.999400
Bullet #900 x: 900.040800 y: 899.999200
Bullet #900 x: 900.040900 y: 899.999100
Bullet #900 x: 900.041000 y: 899.999000
Bullet #900 x: 900.041000 y: 899.999000
Bullet #900 x: 900.041200 y: 899.998800
Bullet #900 x: 900.041300 y: 899.998700
Bullet #900 x: 900.041300 y: 899.998700
Bullet #900 x: 900.041500 y: 899.998500
Bullet #900 x: 900.041600 y: 899.998400
Bullet #900 x: 900.041700 y: 899.998300
Bullet #900 x: 900.041800 y: 899.998200
Bullet #900 x: 900.041900 y: 899.998100
Bullet #900 x: 900.042000 y: 899.998000

*/

	/*
Bullet #900 x: 900.040100 y: 899.999900
Bullet #900 x: 900.040200 y: 899.999800
Bullet #900 x: 900.040300 y: 899.999700
Bullet #900 x: 900.040400 y: 899.999600
Bullet #900 x: 900.040500 y: 899.999500
Bullet #900 x: 900.040600 y: 899.999400
Bullet #900 x: 900.040700 y: 899.999300
Bullet #900 x: 900.040800 y: 899.999200
Bullet #900 x: 900.040900 y: 899.999100
Bullet #900 x: 900.041000 y: 899.999000
Bullet #900 x: 900.041100 y: 899.998900
Bullet #900 x: 900.041200 y: 899.998800
Bullet #900 x: 900.041300 y: 899.998700
Bullet #900 x: 900.041400 y: 899.998600
Bullet #900 x: 900.041500 y: 899.998500
Bullet #900 x: 900.041600 y: 899.998400
Bullet #900 x: 900.041700 y: 899.998300
Bullet #900 x: 900.041800 y: 899.998200
Bullet #900 x: 900.041900 y: 899.998100
Bullet #900 x: 900.042000 y: 899.998000


*/






	// cudaMalloc( (void**)&bd, isize ); 
	// cudaMemcpyAsync( ad, a, csize, cudaMemcpyHostToDevice ); 
	// cudaMemcpyAsync( bd, b, isize, cudaMemcpyHostToDevice ); 
	// 
	// dim3 dimBlock( blocksize, 1 );
	// dim3 dimGrid( 1, 1 );
	// hello<<<dimGrid, dimBlock>>>(ad, bd);
	// cudaMemcpyAsync( a, ad, csize, cudaMemcpyDeviceToHost ); 
	// cudaFree( ad );
	// cudaFree( bd );
	// 
	// printf("%s\n", a);
	// return EXIT_SUCCESS;
}


