// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010
// http://computer-graphics.se/hello-world-for-cuda.html
 
#include <stdio.h>
 
const int block_size = 64; 
const int block_width = 64;
const int block_height = 1;

typedef enum {FORM, ACTIVE, FADE, DEAD} status;
typedef 

 
typedef struct point_t{
    double x;
    double y;
} point;
 
typedef struct bullet_t{
    point position;
    point velocity;
    point acceleration;
    size_t bullet_type;
    int age;
    double w;
    
    
    
} basic_bullet;
 
__device__ void initialize_bullet(basic_bullet* bullet,
    double x, double y, double x_v, double y_v){
    bullet->x = x;
    bullet->y = y;
    bullet->x_v = x_v;
    bullet->y_v = y_v;
}

__global__ void initialize_bullets(basic_bullet* bullets,
                                   size_t bullet_count,
                                   size_t total_threads){
    int bullet_index = threadIdx.x;
    while(bullet_index < bullet_count){
        initialize_bullet(bullets + bullet_index,
        bullet_index + .01*threadIdx.x, bullet_index+.01*threadIdx.y,
        .0001, -.0001);
        bullet_index += total_threads;
    }
}

__device__ void move_bullet(basic_bullet* bullet){
    bullet->x += bullet->x_v;
    bullet->y += bullet->y_v;
}

__global__ void move_bullets(basic_bullet* bullets,
                               size_t bullet_count,
                               size_t total_threads)
{
    int bullet_index = threadIdx.x;
    while(bullet_index < bullet_count){
        move_bullet(bullets + bullet_index);
        bullet_index += total_threads;
    }
}

__device__ void transfer_bullet_position(basic_bullet* bullet,
                                         position* output){
    output->x = bullet->x;
    output->y = bullet->y;
}

__global__ void transfer_bullets_position(basic_bullet* bullets,
                                          position* output,
                                          size_t bullet_count,
                                          size_t total_threads){
    int bullet_index = threadIdx.x;
    while(bullet_index < bullet_count){
        transfer_bullet_position(bullets + bullet_index,
                                 output + bullet_index);
        bullet_index += total_threads;
    }    
}


 
int main()
{
 
    basic_bullet* bullets_d;
    position* positions_d;
    position* positions_h;
    
    
     
    const int bullets_count = 100000;
    const int bullets_size = bullets_count*sizeof(basic_bullet);
    const int positions_size = bullets_count*sizeof(position);
    
    dim3 dimBlock( block_width, block_height );
    dim3 dimGrid( 1, 1 );
 
// printf("%s", a);
// 
    cudaMalloc( (void**)&bullets_d, bullets_size );
    cudaMalloc( (void**)&positions_d, positions_size);
    cudaMallocHost( (void**)&positions_h, positions_size);
    positions_h = (position*) malloc(positions_size);
    
    
                                   
    int bullet_index = 90000;
    
    cudaStream_t stream1, stream2, stream3, stream4;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);
    
    initialize_bullets<<<dimGrid, dimBlock, 0, stream4>>>(
                                   bullets_d,
                                   bullets_count,
                                   block_size);
                                   
    cudaStreamSynchronize(stream4);
        
    for(int i = 0; i < 60; ++i){
        
        
        
        transfer_bullets_position<<<dimGrid, dimBlock, 0, stream2>>>(
                                       bullets_d,
                                       positions_d,
                                       bullets_count,
                                       block_size);
            
                                   
        //cudaDeviceSynchronize();
        
        move_bullets<<<dimGrid, dimBlock, 0, stream1>>>(
                                       bullets_d,
                                       bullets_count,
                                       block_size);
        for(bullet_index = 0; bullet_index < bullets_count; ++bullet_index){
            if(bullet_index==90000){
                printf("Bullet #%d x: %f y: %f \n", 
                bullet_index, 
                positions_h[bullet_index].x,
                positions_h[bullet_index].y);
            }
           
        }
        cudaMemcpyAsync( positions_h, positions_d, 
                     positions_size, cudaMemcpyDeviceToHost, stream3 );
                     
        //cudaStreamSynchronize(stream3);
        
        
           
        
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
    cudaFree(positions_d);
    cudaFreeHost(positions_h);
    
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


