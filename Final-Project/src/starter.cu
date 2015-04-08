// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010
// ://computer-graphics.se/hello-world-for-cuda.html
 
#include <stdio.h>
#include <math.h>
 
const int block_size = 64; 
const int block_width = 64;
const int block_height = 1;

typedef enum {FORM, ACTIVE, FADE, DEAD} bullet_status;
typedef enum {BASIC} bullet_type;

 
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

typedef struct draw_info_t{
    point pos;
    double theta;
    bullet_status status;
    bullet_type type;
} draw_info;

__device__ __host__ point xy_to_rt(point* xy_point){
    double r = hypot(xy_point->x, xy_point->y);
    double t = ((xy_point->x != 0) || (xy_point->y != 0)) ?
               atan2(xy_point -> y, xy_point -> x):
               0;
    point return_value = {r, t};
    return return_value;
}

__device__ __host__ point rt_to_xy(point* rt_point){
    double r = rt_point->x;
    double t = rt_point->y;
    point return_value = {r*cos(t), r*sin(t)};
    return return_value;
}

__global__ void initialize_all_bullet(bullet* bullets,
                               size_t bullet_count,
                               size_t total_threads)
{
    int bullet_index = threadIdx.x;
    while(bullet_index < bullet_count){
        point rt_vector = {bullet_index, bullet_index};
        bullets[bullet_index].pos = rt_to_xy(&rt_vector);
        rt_vector = (point){bullet_index*.01, bullet_index};
        bullets[bullet_index].vel = rt_to_xy(&rt_vector);
        rt_vector = (point){bullet_index*-.0001, bullet_index};
        bullets[bullet_index].acc = rt_to_xy(&rt_vector);
        bullets[bullet_index].theta = rt_vector.y;
        bullets[bullet_index].w = -.001;
        
        bullet_index += total_threads;
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
}

__global__ void update_all_bullet(bullet* bullets,
                               size_t bullet_count,
                               size_t total_threads)
{
    int bullet_index = threadIdx.x;
    while(bullet_index < bullet_count){
        update_bullet(bullets + bullet_index);
        bullet_index += total_threads;
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
                                          size_t bullet_count,
                                          size_t total_threads){
    int bullet_index = threadIdx.x;
    while(bullet_index < bullet_count){
        extract_bullet_draw_info(bullets + bullet_index,
                                 output + bullet_index);
        bullet_index += total_threads;
    }    
}


 
int main()
{
 
    bullet* bullets_d;
    draw_info* draw_infos_d;
    draw_info* draw_infos_h;
    
    
     
    const int bullets_count = 10000;
    const int bullets_size = bullets_count*sizeof(bullet);
    const int draw_infos_size = bullets_count*sizeof(draw_info);
    
    dim3 dimBlock( block_width, block_height );
    dim3 dimGrid( 1, 1 );
 
// printf("%s", a);
// 
    cudaMalloc( (void**)&bullets_d, bullets_size );
    cudaMalloc( (void**)&draw_infos_d, draw_infos_size);
    cudaMallocHost( (void**)&draw_infos_h, draw_infos_size);
    draw_infos_h = (draw_info*) malloc(draw_infos_size);
    
    
                                   
    int bullet_index = 9000;
    
    cudaStream_t stream1, stream2, stream3, stream4;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);
    
    initialize_all_bullet<<<dimGrid, dimBlock, 0, stream4>>>(
                                   bullets_d,
                                   bullets_count,
                                   block_size);
                                   
    cudaStreamSynchronize(stream4);
        
    for(int i = 0; i < 60; ++i){
        
        
        
        extract_all_bullet_draw_info<<<dimGrid, dimBlock, 0, stream2>>>(
                                       bullets_d,
                                       draw_infos_d,
                                       bullets_count,
                                       block_size);
            
                       
        cudaDeviceSynchronize();
        
        update_all_bullet<<<dimGrid, dimBlock, 0, stream1>>>(
                                       bullets_d,
                                       bullets_count,
                                       block_size);
        
        if (cudaSuccess != cudaMemcpyAsync( draw_infos_h, draw_infos_d, 
                     draw_infos_size, cudaMemcpyDeviceToHost, stream3 )){
                     printf("failure \n");
                     }
                     
        cudaDeviceSynchronize();
        
        for(bullet_index = 0; bullet_index < bullets_count; ++bullet_index){
            if(bullet_index == 900){
                printf("Bullet #%d x: %f y: %f t: %f \n", 
                bullet_index, 
                draw_infos_h[bullet_index].pos.x,
                draw_infos_h[bullet_index].pos.y,
                draw_infos_h[bullet_index].theta);
                if(draw_infos_h[bullet_index].pos.x==0){
                    printf("failure2\n");
                }
            }
            
           
        }    
        
        
           
        
    }
                                   
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
    
    
    
                     
    cudaFree(bullets_d);
    cudaFree(draw_infos_d);
    cudaFreeHost(draw_infos_h);
    
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


