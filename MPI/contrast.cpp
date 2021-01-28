#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
#include <mpi.h>
#include <time.h>
#include <math.h>

void run_cpu_color_test(PPM_IMG img_in);
void run_cpu_gray_test(PGM_IMG img_in);

const int root = 0; // Root process

int * sendcounts;
int * displs;
int rem_h = 0;

int main(int argc, char *argv[])
{
    PGM_IMG img_ibuf_g, img_tmp_ibuf_g;
    PPM_IMG img_ibuf_c, img_tmp_ibuf_c;

    int rank, size;
    double start_time, end_time, result_time;
    int  namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv); // Init MPI application
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Who am I
    MPI_Comm_size(MPI_COMM_WORLD, &size); // How many processes
    MPI_Get_processor_name(processor_name, &namelen); // Get processor name

    fprintf(stderr,"Process %d of %d on %s\n", rank, size, processor_name); // Stderr is better than stdout because of stdout need to free using fflush() 

    start_time = MPI_Wtime(); // Start time

    // Root process read images
    if (rank == root)
    {
        fprintf(stderr, "Process %d is reading images...\n", rank);
        img_ibuf_g = read_pgm("in.pgm");
        img_ibuf_c = read_ppm("in.ppm");
    }

#pragma region SendCounts and Displacements

    int rem = (img_ibuf_c.w * img_ibuf_c.h) % size; // Elements remaining after division among processes
    int sum = 0; // Used to calculate displacements

    sendcounts = (int *)malloc(sizeof(int)*size);
    displs = (int *)malloc(sizeof(int)*size);

    // Calculate send counts and displacements
    for (int i = 0; i < size; i++) 
    {
        sendcounts[i] = (img_ibuf_c.w * img_ibuf_c.h) / size;
        
        displs[i] = i * ((img_ibuf_c.w * img_ibuf_c.h) / size);
    }

    sendcounts[size-1] += rem;

#pragma endregion SendCounts and Displacements

#pragma region Imagen_PGM

    // All processes know width and height of the image
    MPI_Bcast(&img_ibuf_g.w, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(&img_ibuf_g.h, 1, MPI_INT, root, MPI_COMM_WORLD);
    
    for(int i=0; i<size; i++)
    {
        MPI_Bcast(&sendcounts[i], 1, MPI_INT, root, MPI_COMM_WORLD);
        MPI_Bcast(&displs[i], 1, MPI_INT, root, MPI_COMM_WORLD);
    }

    img_tmp_ibuf_g.w = img_ibuf_g.w;
    rem_h = img_ibuf_g.h % size;
    img_tmp_ibuf_g.h =  (int)ceil(sendcounts[rank] / (float)img_tmp_ibuf_g.w);

    img_tmp_ibuf_g.img = (unsigned char *)malloc(img_tmp_ibuf_g.w * img_tmp_ibuf_g.h * sizeof(unsigned char));
    
    // Divide the data among processes as described by sendcounts and displacements
    MPI_Scatterv(img_ibuf_g.img, sendcounts, displs, MPI_UNSIGNED_CHAR, img_tmp_ibuf_g.img,
                sendcounts[rank], MPI_UNSIGNED_CHAR, root, MPI_COMM_WORLD);

    if (rank == root) 
    {
        fprintf(stderr, "Running contrast enhancement for gray-scale images...\n");
    }
    run_cpu_gray_test(img_tmp_ibuf_g);

#pragma endregion Imagen_PGM

#pragma region Imagen_PPM

    // All processes know width and height of the image
    MPI_Bcast(&img_ibuf_c.w, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(&img_ibuf_c.h, 1, MPI_INT, root, MPI_COMM_WORLD);

    img_tmp_ibuf_c.w = img_ibuf_c.w;
    img_tmp_ibuf_c.h = (int)ceil(sendcounts[rank] / (float)img_tmp_ibuf_c.w);
    
    img_tmp_ibuf_c.img_b = (unsigned char *) malloc(img_tmp_ibuf_c.w * img_tmp_ibuf_c.h * sizeof(unsigned char));
    img_tmp_ibuf_c.img_g = (unsigned char *) malloc(img_tmp_ibuf_c.w * img_tmp_ibuf_c.h * sizeof(unsigned char));
    img_tmp_ibuf_c.img_r = (unsigned char *) malloc(img_tmp_ibuf_c.w * img_tmp_ibuf_c.h * sizeof(unsigned char));

    // Divide the data among processes as described by sendcounts and displacements
    MPI_Scatterv(img_ibuf_c.img_b, sendcounts, displs, MPI_UNSIGNED_CHAR, img_tmp_ibuf_c.img_b,
                sendcounts[rank], MPI_UNSIGNED_CHAR, root, MPI_COMM_WORLD);

    MPI_Scatterv(img_ibuf_c.img_g, sendcounts, displs, MPI_UNSIGNED_CHAR, img_tmp_ibuf_c.img_g, 
                sendcounts[rank], MPI_UNSIGNED_CHAR, root, MPI_COMM_WORLD);

    MPI_Scatterv(img_ibuf_c.img_r, sendcounts, displs, MPI_UNSIGNED_CHAR, img_tmp_ibuf_c.img_r, 
                sendcounts[rank], MPI_UNSIGNED_CHAR, root, MPI_COMM_WORLD);

    if (rank == root) 
    {
        printf("Running contrast enhancement for HSL and YUV images...\n");
    }
    run_cpu_color_test(img_tmp_ibuf_c);

#pragma endregion Imagen_PPM

	MPI_Barrier(MPI_COMM_WORLD); // Blocks the process until all processes belonging to the specified communicator execute it.
    end_time = MPI_Wtime(); // End time
    result_time = (end_time - start_time); // Result time

    // Root process is the only one that print
    if (rank == root)
    {
        fprintf(stderr, "\nResult time: %f seconds\n", result_time);
        free_pgm(img_ibuf_g);
        free_ppm(img_ibuf_c);
    }

    MPI_Finalize(); // End the MPI application (mandatory)

    return 0;
}

void run_cpu_gray_test(PGM_IMG img_in)
{
    PGM_IMG img_obuf, img_tmp_obuf;
    double start_time, end_time, result_time;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Who am I
    MPI_Comm_size(MPI_COMM_WORLD, &size); // How many processes

    if (rank == root) 
    {
        printf("Starting CPU processing...\n");
    }

    start_time = MPI_Wtime();

    img_tmp_obuf.img = (unsigned char *) malloc(img_in.w * img_in.h * sizeof(unsigned char));
    img_obuf.img = (unsigned char *) malloc(img_in.w * img_in.h * size * sizeof(unsigned char));

    img_tmp_obuf = contrast_enhancement_g(img_in);

    // Gathers into specified locations from all processes in a group
    MPI_Gatherv(img_tmp_obuf.img, sendcounts[rank], MPI_UNSIGNED_CHAR, img_obuf.img, 
                sendcounts, displs, MPI_UNSIGNED_CHAR, root, MPI_COMM_WORLD);
    
	MPI_Barrier(MPI_COMM_WORLD); // Blocks the process until all processes belonging to the specified communicator execute it.
    end_time = MPI_Wtime(); // End of HSL calculation time

    img_obuf.w = img_in.w;
    if(size != 1)
    {
        img_obuf.h = (img_in.h * size) - (size - rem_h);
    }
    else
    {
        img_obuf.h = img_in.h;
    }
    

    if (rank == root)
    {
        printf("Process %d is writting image out.pgm...\n", rank);
        write_pgm(img_obuf, "out.pgm");
        result_time = (end_time - start_time) * 1000;
        printf("Gray image processing time: %f (ms)\n", result_time);
        free_pgm(img_obuf);
        free_pgm(img_tmp_obuf);
    }
}

void run_cpu_color_test(PPM_IMG img_in)
{
    PPM_IMG img_obuf_hsl, img_obuf_yuv;
    PPM_IMG img_tmp_obuf_hsl, img_tmp_obuf_yuv;
    int rank, size;
    double start_hsl_time, end_hsl_time, result_hsl_time;
    double start_yuv_time, end_yuv_time, result_yuv_time;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Who am I
    MPI_Comm_size(MPI_COMM_WORLD, &size); // How many 	processes

    if (rank == root) 
    {
        printf("Starting CPU processing...\n");
    }

    start_hsl_time = MPI_Wtime();

#pragma region HSL

    img_tmp_obuf_hsl.img_b = (unsigned char *) malloc(img_in.h * img_in.w * sizeof(unsigned char));
    img_obuf_hsl.img_b = (unsigned char *) malloc(img_in.h * img_in.w * size * sizeof(unsigned char));

    img_tmp_obuf_hsl.img_g = (unsigned char *) malloc(img_in.h * img_in.w * sizeof(unsigned char));
    img_obuf_hsl.img_g = (unsigned char *) malloc(img_in.h * img_in.w * size * sizeof(unsigned char));

    img_tmp_obuf_hsl.img_r = (unsigned char *) malloc(img_in.h * img_in.w * sizeof(unsigned char));
    img_obuf_hsl.img_r = (unsigned char *) malloc(img_in.h * img_in.w * size * sizeof(unsigned char));

    img_tmp_obuf_hsl = contrast_enhancement_c_hsl(img_in);

    // Gathers into specified locations from all processes in a group
    MPI_Gatherv(img_tmp_obuf_hsl.img_b, sendcounts[rank], MPI_UNSIGNED_CHAR, img_obuf_hsl.img_b, 
                sendcounts, displs, MPI_UNSIGNED_CHAR, root, MPI_COMM_WORLD);

    // Gathers into specified locations from all processes in a group
    MPI_Gatherv(img_tmp_obuf_hsl.img_g, sendcounts[rank], MPI_UNSIGNED_CHAR, img_obuf_hsl.img_g, 
                sendcounts, displs, MPI_UNSIGNED_CHAR, root, MPI_COMM_WORLD);
    
    // Gathers into specified locations from all processes in a group
    MPI_Gatherv(img_tmp_obuf_hsl.img_r, sendcounts[rank], MPI_UNSIGNED_CHAR, img_obuf_hsl.img_r, 
                sendcounts, displs, MPI_UNSIGNED_CHAR, root, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD); // Blocks the process until all processes belonging to the specified communicator execute it.
    end_hsl_time = MPI_Wtime(); // End of HSL calculation time

    img_obuf_hsl.w = img_in.w;
    
    if(size != 1)
    {
        img_obuf_hsl.h = (img_in.h * size) - (size - rem_h);
    }
    else
    {
        img_obuf_hsl.h = img_in.h;
    }

    if (rank == root)
    {        
        printf("Writting image out_hsl.pgm...\n");
        write_ppm(img_obuf_hsl, "out_hsl.ppm");
        result_hsl_time = (end_hsl_time - start_hsl_time) * 1000;
        printf("HSL processing time: %f (ms)\n", result_hsl_time);
        free_ppm(img_obuf_hsl);
        free_ppm(img_tmp_obuf_hsl);
    }

#pragma endregion HSL

#pragma region YUV

    start_yuv_time = MPI_Wtime();
    img_tmp_obuf_yuv.img_b = (unsigned char *) malloc(img_in.h * img_in.w * sizeof(unsigned char));
    img_obuf_yuv.img_b = (unsigned char *) malloc(img_in.h * img_in.w * size * sizeof(unsigned char));

    img_tmp_obuf_yuv.img_g = (unsigned char *) malloc(img_in.h * img_in.w * sizeof(unsigned char));
    img_obuf_yuv.img_g = (unsigned char *) malloc(img_in.h * img_in.w * size * sizeof(unsigned char));

    img_tmp_obuf_yuv.img_r = (unsigned char *) malloc(img_in.h * img_in.w * sizeof(unsigned char));
    img_obuf_yuv.img_r = (unsigned char *) malloc(img_in.h * img_in.w * size * sizeof(unsigned char));

    img_tmp_obuf_yuv = contrast_enhancement_c_yuv(img_in);

    // Gathers into specified locations from all processes in a group
    MPI_Gatherv(img_tmp_obuf_yuv.img_b, sendcounts[rank], MPI_UNSIGNED_CHAR, img_obuf_yuv.img_b, 
                sendcounts, displs, MPI_UNSIGNED_CHAR, root, MPI_COMM_WORLD);

    // Gathers into specified locations from all processes in a group
    MPI_Gatherv(img_tmp_obuf_yuv.img_g, sendcounts[rank], MPI_UNSIGNED_CHAR, img_obuf_yuv.img_g, 
                sendcounts, displs, MPI_UNSIGNED_CHAR, root, MPI_COMM_WORLD);
    
    // Gathers into specified locations from all processes in a group
    MPI_Gatherv(img_tmp_obuf_yuv.img_r, sendcounts[rank], MPI_UNSIGNED_CHAR, img_obuf_yuv.img_r, 
                sendcounts, displs, MPI_UNSIGNED_CHAR, root, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD); // Blocks the process until all processes belonging to the specified communicator execute it.
    end_yuv_time = MPI_Wtime(); // End of YUV calculation time
    
    img_obuf_yuv.w = img_in.w;
    
    if(size != 1)
    {
        img_obuf_yuv.h = (img_in.h * size) - (size - rem_h);
    }
    else
    {
        img_obuf_yuv.h = img_in.h;
    }

    if (rank == root)
    {
        printf("Writting image out_yuv.pgm...\n");
        write_ppm(img_obuf_yuv, "out_yuv.ppm");
        result_yuv_time = (end_yuv_time - start_yuv_time) * 1000;
        printf("YUV processing time: %f (ms)\n", result_yuv_time);
        free_ppm(img_tmp_obuf_yuv);
        free_ppm(img_obuf_yuv);
    }

#pragma endregion YUV
}

PPM_IMG read_ppm(const char * path){
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
    printf("Image size ppm: %d x %d\n", result.w, result.h);


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

void write_ppm(PPM_IMG img, const char * path){
    FILE * out_file;
    int i;

    char * obuf = (char *)malloc(3 * img.w * img.h * sizeof(char));

    for(i = 0; i < img.w*img.h; i ++){
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

PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];


    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }

    fscanf(in_file, "%s", sbuf); /*Skip the magic number*/
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    int img_size = result.w * result.h;
    printf("Image size pgm: %d x %d\n", result.w, result.h);


    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));


    fread(result.img, sizeof(unsigned char), result.w*result.h, in_file);
    fclose(in_file);

    return result;
}

void write_pgm(PGM_IMG img, const char * path){
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

