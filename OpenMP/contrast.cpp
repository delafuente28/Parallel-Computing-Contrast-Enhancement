#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
#include <time.h>
#include <omp.h>

void run_cpu_color_test(PPM_IMG img_in);
void run_cpu_gray_test(PGM_IMG img_in);

// export OMP_NUM_THREADS=8 // Environment variable that sets the number of threads.

int main(int argc, char **argv)
{
    int nthreads;
    #pragma omp parallel
    {
        nthreads = omp_get_num_threads(); // Get number of threads
    }

    PGM_IMG img_ibuf_g;
    PPM_IMG img_ibuf_c;

    img_ibuf_g = read_pgm("in.pgm"); // Read gray image
    img_ibuf_c = read_ppm("in.ppm"); // Read color image
    
    printf("Running contrast enhancement for gray-scale images with %d threads.\n", nthreads);
    run_cpu_gray_test(img_ibuf_g); // Compute gray image in sequential mode --> 500ms
    free_pgm(img_ibuf_g); // Free buffer
   
    printf("Running contrast enhancement for color images with %d threads.\n", nthreads);
    run_cpu_color_test(img_ibuf_c); // Compute color image in sequential mode --> 7700ms HSL / 3500ms YUV
    free_ppm(img_ibuf_c); // Free buffer
    
    return 0;
}

void run_cpu_color_test(PPM_IMG img_in)
{
    double start_hsl, end_hsl, result_time_hsl;
    double start_yuv, end_yuv, result_time_yuv;
    PPM_IMG img_obuf_hsl, img_obuf_yuv;
    
    printf("Starting CPU processing...\n");
    
    start_hsl = omp_get_wtime(); // HSL start time
    img_obuf_hsl = contrast_enhancement_c_hsl(img_in); // Run HSL contrast enhancement
    end_hsl = omp_get_wtime(); // HSL end time

    result_time_hsl = end_hsl - start_hsl; // HSL result time

        printf("HSL processing time: %lf (ms)\n", result_time_hsl * 1000.0);
        write_ppm(img_obuf_hsl, "out_hsl.ppm");
        free_ppm(img_obuf_hsl);
    

    start_yuv = omp_get_wtime(); // YUV start time
    img_obuf_yuv = contrast_enhancement_c_yuv(img_in); // Run YUV contrast enhancement
    end_yuv = omp_get_wtime(); // YUV end time

    result_time_yuv = end_yuv - start_yuv; // YUV result time
    
        printf("YUV processing time: %lf (ms)\n", result_time_yuv * 1000.0);
        write_ppm(img_obuf_yuv, "out_yuv.ppm");
        free_ppm(img_obuf_yuv);
    
}

void run_cpu_gray_test(PGM_IMG img_in)
{
    PGM_IMG img_obuf;
    double start, end, result_time;
    
    printf("Starting CPU processing...\n");

    start = omp_get_wtime(); // Get start time
    img_obuf = contrast_enhancement_g(img_in); // Run gray contrast enhancement
    end = omp_get_wtime(); // Get end time

    result_time = end - start; // Get result time
    

        printf("Processing time: %lf (ms)\n", result_time * 1000.0);
        write_pgm(img_obuf, "out.pgm");
        free_pgm(img_obuf);
    
}

PPM_IMG read_ppm(const char * path)
{
    FILE * in_file;
    char sbuf[256];
    
    char *ibuf;
    PPM_IMG result;
    int v_max, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    /*Skip the magic number*/
    fscanf(in_file, "%s", sbuf);


    //result = malloc(sizeof(PPM_IMG));
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("PPM Image size: %d x %d\n", result.w, result.h);
    

    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    ibuf         = (char *)malloc(3 * result.w * result.h * sizeof(char));

    
    fread(ibuf,sizeof(unsigned char), 3 * result.w*result.h, in_file);

    for(i = 0; i < result.w*result.h; i ++){
        result.img_r[i] = ibuf[3*i + 0];
        result.img_g[i] = ibuf[3*i + 1];
        result.img_b[i] = ibuf[3*i + 2];
    }
    
    fclose(in_file);
    free(ibuf);
    
    return result;
}

void write_ppm(PPM_IMG img, const char * path)
{
    FILE * out_file;
    int i;
    
    char * obuf = (char *)malloc(3 * img.w * img.h * sizeof(char));

    for(i = 0; i < img.w*img.h; i++)
    {
        obuf[3*i + 0] = img.img_r[i];
        obuf[3*i + 1] = img.img_g[i];
        obuf[3*i + 2] = img.img_b[i];
    }

    out_file = fopen(path, "wb");
    fprintf(out_file, "P6\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(obuf,sizeof(unsigned char), 3*img.w*img.h, out_file);
    fclose(out_file);
    free(obuf);
}

void free_ppm(PPM_IMG img)
{
    free(img.img_r);
    free(img.img_g);
    free(img.img_b);
}

PGM_IMG read_pgm(const char * path)
{
    FILE * in_file;
    char sbuf[256];    
    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "r");
    
    if (in_file == NULL)
    {
        printf("Input file not found!\n");
        exit(1);
    }
    
    fscanf(in_file, "%s", sbuf); /*Skip the magic number*/
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("PGM Image size: %d x %d\n", result.w, result.h);
    
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    
    fread(result.img,sizeof(unsigned char), result.w*result.h, in_file);    
    fclose(in_file);
    
    return result;
}

void write_pgm(PGM_IMG img, const char * path)
{
    FILE * out_file;
    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(img.img,sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

void free_pgm(PGM_IMG img)
{
    free(img.img);
}