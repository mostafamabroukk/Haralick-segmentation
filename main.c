/* Code for Chan-Vese segmentation */
/* Copyright Levi Valgaerts 2013   */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <limits.h>
#include <time.h>
#include <GL/glut.h>

#include "libs/alloc_mem_linear_mult.h"
#include "libs/io_lib.h"
#include "libs/tensor_lib.h"
#include "libs/matrix_lib.h"
#include "libs/funct_lib.h"
#include "libs/eigen_lib.h"
#include "libs/color_lib.h"
#include "libs/name_lib.h"
#include "libs/bounds_lib.h"
#include "libs/psi_lib.h"
#include <dirent.h>

/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */
#define PI          3.14159265
#define BIGNUMBER   10000000000000.0
#define BANDWIDTH   20.0
/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */

/* maximum length of a character array */
#define MAXLINE 100

int i, j;
int help; 															/* helper variable */
char title[MAXLINE]; 												/* the title of the OpenGL window */
char image_full_path_str[MAXLINE]; 											/* name of the image */
char image_title_str[MAXLINE]; 											/* name of the image */

char image_chan_2_str[MAXLINE];
char image_chan_1_str[MAXLINE];
char image_chan_3_str[MAXLINE];
char image_chan_4_str[MAXLINE];

char image_OT_str[MAXLINE];

//char out_name[MAXLINE] = "out.pgm";   							/* output image name without extension */
char image_published_str[MAXLINE];

/* the types of level set initialisation */
enum init_mode {
	SQUARE, GRID, CIRCLE, PRESET, init_end
};
const char *init_modes[] = { "SQUARE", "GRID", "CIRCLE", "PRESET" };
int init_type;

/* the types of segmentation */
enum seg_mode {
	CV_, CV_VEC_2, CV_VEC_3, CV_VEC_4, seg_end
};
const char *seg_modes[] = { "CV", "CV_vec_2", "CV_vec_3", "CV_vec_4" };
int seg_type;

int active_param; /* active parameter */
double max_value, min_value, temp_value;

int nx, ny, bx, by;
double hx, hy;
double epsilon;										 /* epsilon diffusivity (GAC) / Heaviside function (CV) */
double sigma;										 /* presmoothing */
double tau; 										 /* time step */
double T;											 /* stopping time */
double T_incr; 										 /* time increment */
double nu; 											 /* weight area constraint */
double mu; 											 /* weight length constraint */

GLubyte *pixels; 									/* pixel aray for Open-GL */
double **f_post_seg;								/* post segmented image */
double **f_ot;										/* GROUND TRUTH */
double **f_c1;										/* image channel 1 */
double **f_c1_s;									/* smooth image channel 1 */
double **f_c2; 										/* image channel 1 */
double **f_c2_s;									/* smooth image channel 1 */
double **f_c3; 										/* image channel 1 */
double **f_c3_s; 									/* smooth image channel 1 */
double **f_c4; 										/* image channel 4 */
double **f_c4_s; 									/* smooth image channel 4 */
double **f2; 										/* image with useful edges */
double **f2_s; 										/* smooth image */
double **f_work; 									/* working copy for MCM */
double **u, **v; 									/* optical flow field */
double **u_s, **v_s; 								/* smoothed optical flow field */
double **out;										/* output */
double **phi; 										/* level set function */
double **phi_stored; 								/* stored level set function for initialisation */
double ***p6; 										/* P6 image */

double** GLCmatrix_main;
double final_c11, final_c12, final_c21, final_c22;

double lambda11, lambda12, lambda21, lambda22;
double **phi1, **phi2;
double **phi_stored1, **phi_stored2;

bool OT_exists = true;
bool draw;
bool publish_segmented_image;
double average_score = 0;
double global_f_measure = 0;
double old_global_f_measure = 0;

double tumor_gray_level = 96;
double small_details_gray_level = 255;
double background_gray_level = 0;
double skull_gray_level = 160;


// Stopping Criteria
//==================================================
double maximumNoIterations = 100000;
double minimumDifference = 0.00001;

bool regular_evaluation= true;
int regular_evaluation_int=20;
bool boolean_of_colors_hard_coded = true;
bool use_median_filtering = false;
//==================================================

inline double max(double a, double b) {
	return (a > b) ? a : b;
}

inline double min(double a, double b) {
	return (a < b) ? a : b;
}

/* computes the Heaviside function */
inline double heaviside_function(double z, /* argument */ double epsilon /* regularisation parameter */)
{
	//return (phi >= 0.0)?1.0:0.0;
	return 0.5 * (1 + (2 / PI) * atan(z / epsilon));
}

/* computes the derivative of the heaviside function */
inline double delta_function(double z, /* argument */ double epsilon /* regularisation parameter */)
{
	return epsilon / (PI * (epsilon * epsilon + z * z));
}

/* draws the image in OpenGL */
void draw_image(double ***p6, 							/* out : RGB image */
int format, 											/* in  : 0 : p5 / 1 : p6 */
GLubyte *pixels,										/* use : display array */
int magnify, 											/* in  : scaling factor */
int nx,													/* in  : size in x-direction */
int ny, 												/* in  : size in y-direction */
int bx, 												/* in  : boundary size in x-direction */
int by 													/* in  : boundary size in y-direction */
);

/* writes the level set function as a black and white image */
void level_set_to_image(double **phi1, double **phi2, int nx, /* in  : size in x-direction */
int ny, /* in  : size in y-direction */
int bx, /* in  : boundary size in x-direction */
int by, /* in  : boundary size in y-direction */
double **image /* out : the gray scale image */
);

/* converts the level set function to a binary array */
void level_set_to_binary(double **phi, /* in  : the level set function */
int nx, /* in  : size in x-direction */
int ny, /* in  : size in y-direction */
int bx, /* in  : boundary size in x-direction */
int by, /* in  : boundary size in y-direction */
double **bin /* out : binary array */
);

/* scales an array to a greyscale image */
void array_to_image2(double **array, /* in  : an array of positive values */
int nx, /* in  : size in x-direction */
int ny, /* in  : size in y-direction */
int bx, /* in  : boundary size in x-direction */
int by, /* in  : boundary size in y-direction */
double **image /* out : the gray scale image */
);

/* initialises the level set function */
void init_level_set(double **f, /* in+out  : the level set function */
int nx, /* in  : size in x-direction */
int ny, /* in  : size in y-direction */
int bx, /* in  : boundary size in x-direction */
int by, /* in  : boundary size in y-direction */
double hx, /* in  : the grid size in x-dir. */
double hy, /* in  : the grid size in y-dir. */
int init_type /* in  : the type of level set initialisation */
);

/* Computes the signed distance function by solving an evolution equation.
 Implementation follows Sussman, Smereka, Osher / Chan, Vese. */
void reinit_level_set(double **phi, /* in+out  : the level set function */
int nx, /* in  : size in x-direction */
int ny, /* in  : size in y-direction */
int bx, /* in  : boundary size in x-direction */
int by, /* in  : boundary size in y-direction */
double hx, /* in  : the grid size in x-dir. */
double hy, /* in  : the grid size in y-dir. */
int iter /* in  : number of iterations */
);

/* Computes the signed distance function to a zero level set. 
 The input is a level set function phi with phi>0.0 within the
 region encircled by the zero level set and phi<0.0 outside the
 zero level set. The output is a level set function which has as
 values the signed distance to the zero level set. */
void signed_distance(double **f, /* in+out  : a level set function */
int nx, /* in  : size in x-direction */
int ny, /* in  : size in y-direction */
int bx, /* in  : boundary size in x-direction */
int by /* in  : boundary size in y-direction */
);

/* Computes the Euclidean Distance Transform of a binary image.
 The input image has to be 0.0 in the foreground and -inf in the background.
 The background will be filled in with the distance to the foreground.
 */
void EDT(double **ff, /* in+out  : binary image, foreground = 0.0, background = -inf */
int nx, /* in  : size in x-direction */
int ny, /* in  : size in y-direction */
int bx, /* in  : boundary size in x-direction */
int by /* in  : boundary size in y-direction */
);


/* segments an image in two regions using a level set function in 
 the Chan - Vese framework */
void CV(double **f, /* in  : the image  */
double **phi1, double **phi2, int nx, /* in  : size in x-direction */
int ny, /* in  : size in y-direction */
int bx, /* in  : boundary size in x-direction */
int by, /* in  : boundary size in y-direction */
double hx, /* in  : the grid size in x-dir. */
double hy, /* in  : the grid size in y-dir. */
double tau, /* in  : time step size */
double T, /* in  : stopping time */
double mu, /* in  : weighting parameter for MCM */
double nu, /* in  : weighting parameter for the area inside the curve */
double epsilon, /* in  : regularisation parameter for the Heaviside function */
double lambda11, double lambda12, double lambda21, double lambda22);

void CV_vec_two_channels(
double **f_c1,
double **f_c2,
double **phi1, double **phi2,
int nx, int ny, int bx, int by,
double hx, double hy, double tau, double T,
double mu, double nu, double epsilon,
double lambda11, double lambda12, double lambda21, double lambda22);

/* segments a vector-valued image in two regions using a level set function in
 the Chan - Vese framework */
void CV_vec_three_channels(double **f_c1, /* in  : the image channel 1 */
double **f_c2, /* in  : the image channel 2 */
double **f_c3, /* in  : the image channel 3 */
double **phi1, double **phi2, int nx, /* in  : size in x-direction */
int ny, /* in  : size in y-direction */
int bx, /* in  : boundary size in x-direction */
int by, /* in  : boundary size in y-direction */
double hx, /* in  : the grid size in x-dir. */
double hy, /* in  : the grid size in y-dir. */
double tau, /* in  : time step size */
double T, /* in  : stopping time */
double mu, /* in  : weighting parameter for MCM */
double nu, /* in  : weighting parameter for the area inside the curve */
double epsilon, /* in  : regularisation parameter for the Heaviside function */
double lambda11, double lambda12, double lambda21, double lambda22);

void CV_vec_four_channels(
double **f_c1,
double **f_c2,
double **f_c3,
double **f_c4,
double **phi1, double **phi2,
int nx, int ny, int bx, int by,
double hx, double hy, double tau, double T,
double mu, double nu, double epsilon,
double lambda11, double lambda12, double lambda21, double lambda22);

void post_segmented_image(double **segmented_img, int nx, int ny, int bx,int by);
void evaluate(double **segmented_img, double **ground_truth, int nx, int ny,int bx, int by);

void LVD(double** input_image, int nx, int ny, int bx, int by, double** output_image);
void sort_array(double array[]);

void median_filter(double** input_image, int nx, int ny, int bx, int by, double** output_image);
double return_median_value (double array[9]);
void extract_textural_features(double** input_image, int nx, int ny, int bx, int by, int counter_of_text_feature, int theta, int window_size);
double** LBP(double** input_image, int nx, int ny, int bx, int by);
void new_images_minus(double** f, double** d, double** t1, double** t2, int nx, int ny, int bx, int by);
void new_images_plus(double** f, double** d, double** t1, double** t2, int nx, int ny, int bx, int by);



/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/

void handleCompute() {

	/* presmooth the image channels */
	// performs convolution with Gaussian with standard deviation sigma
	presmooth_2d(f_c1, f_c1_s, nx, ny, bx, by, hx, hy, sigma, sigma);
	presmooth_2d(f_c2, f_c2_s, nx, ny, bx, by, hx, hy, sigma, sigma);
	presmooth_2d(f_c3, f_c3_s, nx, ny, bx, by, hx, hy, sigma, sigma);
	presmooth_2d(f_c4, f_c4_s, nx, ny, bx, by, hx, hy, sigma, sigma);


//	 output_image_after_median_filter
//	 --------------------------------------------------------------------------
	double** output_image_after_median_filter;
	if(use_median_filtering)
	{
		 ALLOC_MATRIX(1, nx + 2 * bx, ny + 2 * by, &output_image_after_median_filter);
		 median_filter(f_c1, nx, ny, bx, by, output_image_after_median_filter);
		 write_pgm_blank_header("output_image_after_median_filter.pgm", nx, ny);
		 write_pgm_data("output_image_after_median_filter.pgm", output_image_after_median_filter, nx, ny, bx, by);
	}

//	----------------------------------------------------------------------
//	extract_textural_features
// ----------------------------------------------------------------------
//	for (int counter = 1; counter <= 13; counter++)
//	{
//		for (int theta = 0; theta < 150; theta = theta + 45)
//		{
//			extract_textural_features(f_c1, nx, ny, bx, by, counter, theta);
//		}
//	}

//	F6 Sum Of Average
//	F7 Sum Of Variance

//	extract_textural_features(output_image_after_median_filter, nx, ny, bx, by, 7, 0, 7);
//	extract_textural_features(f_c1, nx, ny, bx, by, 6, 0, 7);
//	extract_textural_features(f_c1, nx, ny, bx, by, 6, 45, 7);
//	extract_textural_features(f_c1, nx, ny, bx, by, 6, 90, 7);
//	extract_textural_features(f_c1, nx, ny, bx, by, 6, 135, 7);
//
//
//	extract_textural_features(f_c1, nx, ny, bx, by, 7, 0, 3);
//	extract_textural_features(f_c1, nx, ny, bx, by, 7, 45, 7);
//	extract_textural_features(f_c1, nx, ny, bx, by, 7, 90, 7);
//	extract_textural_features(f_c1, nx, ny, bx, by, 7, 135, 7);
//	exit(0);
//	----------------------------------------------------------------------

//	LBP(f_c1, nx, ny, bx, by);
	exit(0);


//	new_images_plus(f_c1, f_c2, f_c3, f_c4, nx, ny, bx, by);
//	exit(0);


	switch (seg_type) {

	case CV_:        // Chan Vese model

		/* initialise the level set function */
		init_level_set(phi1, nx, ny, bx, by, hx, hy, (int) SQUARE);
		init_level_set(phi2, nx, ny, bx, by, hx, hy, (int) CIRCLE);

		if(use_median_filtering)
			CV(output_image_after_median_filter, phi1, phi2, nx, ny, bx, by, hx, hy, tau, T, mu, nu, epsilon,lambda11, lambda12, lambda21, lambda22);
		else
			CV(f_c1, phi1, phi2, nx, ny, bx, by, hx, hy, tau, T, mu, nu, epsilon,lambda11, lambda12, lambda21, lambda22);
		level_set_to_image(phi1, phi2, nx, ny, bx, by, out);
		break;

	case CV_VEC_2:     // Chan Vese model for vector-valued images

		init_level_set(phi1, nx, ny, bx, by, hx, hy, (int) SQUARE);
		init_level_set(phi2, nx, ny, bx, by, hx, hy, (int) CIRCLE);

		CV_vec_two_channels(f_c1_s, f_c2_s, phi1, phi2, nx, ny, bx, by, hx,hy, tau, T, mu, nu, epsilon, lambda11, lambda12, lambda21, lambda22);
		level_set_to_image(phi1, phi2, nx, ny, bx, by, out);
		break;

	case CV_VEC_3:     // Chan Vese model for vector-valued images

		/* initialise the level set function */
		init_level_set(phi1, nx, ny, bx, by, hx, hy, (int) SQUARE);
		init_level_set(phi2, nx, ny, bx, by, hx, hy, (int) CIRCLE);

		CV_vec_three_channels(f_c1_s, f_c2_s, f_c3_s, phi1, phi2, nx, ny, bx, by, hx,hy, tau, T, mu, nu, epsilon, lambda11, lambda12, lambda21, lambda22);
		level_set_to_image(phi1, phi2, nx, ny, bx, by, out);
		break;

	case CV_VEC_4:     // Chan Vese model for vector-valued images

		/* initialise the level set function */
		init_level_set(phi1, nx, ny, bx, by, hx, hy, (int) SQUARE);
		init_level_set(phi2, nx, ny, bx, by, hx, hy, (int) CIRCLE);

		CV_vec_four_channels(f_c1_s, f_c2_s, f_c3_s, f_c4_s, phi1, phi2, nx, ny, bx, by, hx,hy, tau, T, mu, nu, epsilon, lambda11, lambda12, lambda21, lambda22);
		level_set_to_image(phi1, phi2, nx, ny, bx, by, out);
		break;

	default:break;
	}

	if (OT_exists) {
		post_segmented_image(out, nx, ny, bx, by);
		evaluate(f_post_seg, f_ot, nx, ny, bx, by);

		char post_segmented_image_name[MAXLINE];
		strcpy(post_segmented_image_name, image_title_str);
		strcat(post_segmented_image_name, "_post_segmented_image.pgm");
		write_pgm_blank_header(post_segmented_image_name, nx, ny);
		write_pgm_data(post_segmented_image_name, f_post_seg, nx, ny, bx, by);
	}

	if (publish_segmented_image) {
		write_pgm_blank_header(image_published_str, nx, ny);
		write_pgm_data(image_published_str, out, nx, ny, bx, by);
	}

}

/*----------------------------------------------------------------------------*/

void handleComputeNothing() {
}

/*----------------------------------------------------------------------------*/

void draw_image(double ***p6, /* out : RGB image */
int format, /* in  : 0 : p5 (PGM) / 1 : p6 (PPM) */
GLubyte *pixels, /* use : display array */
int magnify, /* in  : scaling factor */
int nx, /* in  : size in x-direction */
int ny, /* in  : size in y-direction */
int bx, /* in  : boundary size in x-direction */
int by /* in  : boundary size in y-direction */
)

/* visualises an image with Open-GL */
/* this one works for combine = 1,2,3 */

{
	int counter; /* pixel index counter */
	int i, j; /* loop variable */

	/* set pixel counter zero */
	counter = 0;

	/* prepare Open-GL */
	glViewport(0, 0, nx, ny);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, nx, 0, ny, -1.0, 1.0);
	glMatrixMode(GL_MODELVIEW);
	glDisable(GL_DITHER);
	glPixelZoom((GLfloat) magnify, (GLfloat) magnify);

	/* draw pixels */
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(0.0, 0.0, 0.0, 1.0);

	for (i = by; i < ny + by; i++)
	{
		for (j = bx; j < nx + bx; j++)
		{

			pixels[counter++] = (GLubyte) (p6[j][ny + 2 * by - 1 - i][0]);
			pixels[counter++] = (GLubyte) (p6[j][ny + 2 * by - 1 - i][1]);
			pixels[counter++] = (GLubyte) (p6[j][ny + 2 * by - 1 - i][2]);

			//############################################################
//			pixels[counter++] = (GLubyte) (p6[j][ny + 2 * by - 1 - i][3]);
		}

		glRasterPos3f(0, i - by, 0.0);
		glDrawPixels(nx, 1, GL_RGB, GL_UNSIGNED_BYTE, &pixels[(i - by) * 3 * nx]);

		//####################################################################
//		glRasterPos4f(0, i - by, 0.0, 0.0);
//		glDrawPixels(nx, 1, GL_RGBA, GL_UNSIGNED_BYTE,&pixels[(i - by) * 4 * nx]);
	}

	/* swap draw and display buffer */
	glutSwapBuffers();

	return;
}

/*----------------------------------------------------------------------------*/

void handleDraw() {
	/* copy in p6 */
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++) {

			/* display smoothed image */
			/* 	    p6[i][j][0] = byte_range(f_c1_s[i][j]); */
			/* 	    p6[i][j][1] = byte_range(f_c2_s[i][j]); */
			/* 	    p6[i][j][2] = byte_range(f_c3_s[i][j]); */

			/* display unsmoothed image */
			p6[i][j][0] = byte_range(f_c1[i][j]);
			p6[i][j][1] = byte_range(f_c2[i][j]);
			p6[i][j][2] = byte_range(f_c3[i][j]);
//			p6[i][j][3] = byte_range(f_c4[i][j]);

			p6[i + nx][j][0] = byte_range(out[i][j]);
			p6[i + nx][j][1] = byte_range(out[i][j]);
			p6[i + nx][j][2] = byte_range(out[i][j]);
//			p6[i + nx][j][3] = byte_range(out[i][j]);
		}

	/* draw flowfield */
	draw_image(p6, 1, pixels, 1, 2 * nx, ny, bx, by);

	glutPostRedisplay();
}

/*----------------------------------------------------------------------------*/

void showParams() {
	print_console_header("Image Segmentation");

	if (active_param == 1)
		printf("\n (t) (time step)                                    %4.6lf",
				(double) tau);
	else
		printf("\n (t)  time step                                     %4.6lf",
				(double) tau);

	if (active_param == 2)
		printf("\n (T) (time)                                         %4.3lf",
				(double) T);
	else
		printf("\n (T)  time                                          %4.3lf",
				(double) T);

	if (active_param == 7)
		printf("\n (i) (time incr)                                    %4.3lf",
				(double) T_incr);
	else
		printf("\n (i)  time incr                                     %4.3lf",
				(double) T_incr);

	if (active_param == 3)
		printf("\n (e) (epsilon)                                      %4.3lf",
				(double) epsilon);
	else
		printf("\n (e)  epsilon                                       %4.3lf",
				(double) epsilon);

	if (active_param == 4)
		printf("\n (p) (sigma)                                        %4.3lf",
				(double) sigma);
	else
		printf("\n (p)  sigma                                         %4.3lf",
				(double) sigma);

	if (active_param == 5)
		printf("\n (n) (nu)                                           %4.3lf",
				(double) nu);
	else
		printf("\n (n)  nu                                            %4.3lf",
				(double) nu);

	if (active_param == 6)
		printf("\n (m) (mu)                                           %4.3lf",
				(double) mu);
	else
		printf("\n (m)  mu                                            %4.3lf",
				(double) mu);

	if (active_param == 9)
		printf("\n (s) (Segmentation)                                 %s",
				seg_modes[seg_type]);
	else
		printf("\n (s)  Segmentation                                  %s",
				seg_modes[seg_type]);

	print_console_line();
	printf("\n");
	fflush(stdout);
}

/*----------------------------------------------------------------------------*/

void handleKeyboardspecial(int key, int x, int y) {
	switch (key) {

	case GLUT_KEY_F3: /* standard settings */

		tau = 1;     //0.25;
		T = 20.0;
		epsilon = 0.1;     // for diffusivity in GAC or Heaviside function in CV
		sigma = 1.0;
		nu = 0.0; // best choose 0.0 for GAC and needs to be big for CV to give noticable effects
		mu = 50.0;        //1;       // MCM weight in CV

		break;

	case GLUT_KEY_F4:

		/* store the last solution */
		copy_matrix_2d(phi, phi_stored, nx, ny, bx, by);
		printf("\nsolution stored\n");

		break;

	case GLUT_KEY_F5: /* write out image */

		write_pgm_blank_header(image_published_str, nx, ny);
		write_pgm_data(image_published_str, out, nx, ny, bx, by);

		break;

	case GLUT_KEY_DOWN:

		if (active_param == 1) {
			tau -= 0.025;
			if (tau < 0.0)
				tau = 0.0;
			break;
		}
		if (active_param == 2) {
			T -= T_incr;
			if (T < 0.0)
				T = 0.0;
			break;
		}
		if (active_param == 3) {
			epsilon /= sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(10.0))))));
			break;
		}
		if (active_param == 4) {
			sigma -= 0.025;
			if (sigma < 0.3)
				sigma = 0.0;
			break;
		}
		if (active_param == 5) {
			nu /= sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(10.0))))));
			if (fabs(nu) < 0.001)
				nu = 0.0;
			break;
		}
		if (active_param == 6) {
			mu /= sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(10.0))))));
			break;
		}
		if (active_param == 7) {
			T_incr /= 10.0;
			if (T_incr < 1.0)
				T_incr = 1.0;
			break;
		}
		if (active_param == 8) {
			init_type -= 1;
			if (init_type < 0)
				init_type = init_end - 1;
			break;
		}

	case GLUT_KEY_UP:

		if (active_param == 1) {
			tau += 0.025;
			break;
		}
		if (active_param == 2) {
			T += T_incr;
			break;
		}
		if (active_param == 3) {
			epsilon *= sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(10.0))))));
			break;
		}
		if (active_param == 4) {
			sigma += 0.025;
			break;
		}
		if (active_param == 5) {
			nu *= sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(10.0))))));
			if (nu == 0.0)
				nu = 0.001;
			break;
		}
		if (active_param == 6) {
			mu *= sqrt(sqrt(sqrt(sqrt(sqrt(sqrt(10.0))))));
			break;
		}
		if (active_param == 7) {
			T_incr *= 10.0;
			break;
		}
		if (active_param == 8) {
			init_type += 1;
			if (init_type >= init_end)
				init_type = 0;
			break;
		}

	default:
		printf("\nUnknown key pressed (Code %d).", key);
	}
	showParams();

	return;
}

/*----------------------------------------------------------------------------*/

void handleKeyboard(unsigned char key, int x, int y) {
	switch (key) {

	case 't':  // time step
		active_param = 1;
		break;

	case 'T':  // time
		active_param = 2;
		break;

	case 'e':  // epsilon
		active_param = 3;
		break;

	case 'p':  // sigma
		active_param = 4;
		break;

	case 'n':  // nu
		active_param = 5;
		break;

	case 'm':  // mu
		active_param = 6;
		break;

	case 'i':  // time increment
		active_param = 7;
		break;

	case 's':  // segmentation type
		active_param = 9;
		break;

	case 'l':  // lambda
		if (active_param != 10)
			active_param = 10;
		else
			active_param = 11;
		break;

	case '-':  // invert nu
		nu = -nu;
		break;

	case 46:  //.
		handleCompute();
		break;

	default:
		printf("\nUnknown key pressed (Code %d).", key);
	}
	showParams();

	return;
}

/*----------------------------------------------------------------------------*/

void handleMouse(int button, int state, int cx, int cy) {
	printf("\nNo mouse handler yet.");
}

/*----------------------------------------------------------------------------*/
/*----------------------------------------------------------------------------*/

/* -- should be in seg_lib.c: ----------------------------------------------- */

void level_set_to_image(double **phi1, double **phi2, int nx, /* in  : size in x-direction */
int ny, /* in  : size in y-direction */
int bx, /* in  : boundary size in x-direction */
int by, /* in  : boundary size in y-direction */
double **image /* out : the gray scale image */
) {
	int i, j; /* loop variables */
//	double H1 = heaviside_function(phi1[i][j], 0.1);
//	double H2 = heaviside_function(phi2[i][j], 0.1);

	int cnt1 = 0;
	int cnt2 = 0;
	int cnt3 = 0;
	int cnt4 = 0;
	for (i = bx; i < nx + bx; i++) {
		for (j = by; j < ny + by; j++) {
			if (phi1[i][j] > 0.0 && phi2[i][j] > 0.0) {
				cnt1++;
//				image[i][j] = 255.0;
				image[i][j] = final_c11;
			} else if (phi1[i][j] > 0.0 && phi2[i][j] < 0.0) {
				cnt2++;
//				image[i][j] = 128.0;
				image[i][j] = final_c12;
			} else if (phi1[i][j] < 0.0 && phi2[i][j] > 0.0) {
				cnt3++;
//				image[i][j] = 80.0;
				image[i][j] = final_c21;
			} else if (phi1[i][j] < 0.0 && phi2[i][j] < 0.0) {
				cnt4++;
//				image[i][j] = 20.0;
				image[i][j] = final_c22;
			}
		}
	}

//	printf("%d %d %d %d\n", cnt1, cnt2, cnt3, cnt4);
}

void level_set_to_image_final(double **phi1, double **phi2, int nx, /* in  : size in x-direction */
int ny, /* in  : size in y-direction */
int bx, /* in  : boundary size in x-direction */
int by, /* in  : boundary size in y-direction */
double **image /* out : the gray scale image */
) {
	int i, j; /* loop variables */
//	double H1 = heaviside_function(phi1[i][j], 0.1);
//	double H2 = heaviside_function(phi2[i][j], 0.1);

	int cnt = 0;
	for (i = bx; i < nx + bx; i++) {
		for (j = by; j < ny + by; j++) {
			if (phi1[i][j] > 0.0 && phi2[i][j] > 0.0) {
//				image[i][j] = 255.0;
				image[i][j] = final_c11;
			} else if (phi1[i][j] > 0.0 && phi2[i][j] < 0.0) {
//				image[i][j] = 128.0;
				image[i][j] = final_c12;
			} else if (phi1[i][j] < 0.0 && phi2[i][j] > 0.0) {
//				image[i][j] = 80.0;
				image[i][j] = final_c21;
			} else if (phi1[i][j] < 0.0 && phi2[i][j] < 0.0) {
//				image[i][j] = 20.0;
				image[i][j] = final_c22;
			}
		}
	}
}
/*----------------------------------------------------------------------------*/

void level_set_to_binary(double **phi, /* in  : the level set function */
int nx, /* in  : size in x-direction */
int ny, /* in  : size in y-direction */
int bx, /* in  : boundary size in x-direction */
int by, /* in  : boundary size in y-direction */
double **bin /* out : binary array */
) {
	int i, j; /* loop variables */

	/* 1 is bigger than zero, 0 is smaller than zero */
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++) {
			bin[i][j] = (phi[i][j] > 0.0) ? 1.0 : 0.0;
		}
}

/*----------------------------------------------------------------------------*/

void array_to_image2(double **array, /* in  : an array of positive values */
int nx, /* in  : size in x-direction */
int ny, /* in  : size in y-direction */
int bx, /* in  : boundary size in x-direction */
int by, /* in  : boundary size in y-direction */
double **image /* out : the gray scale image */
) {
	int i, j; /* loop variables */
	double max; /* maximum value in the array */
	double factor; /* scaling factor */

	max = 0.0;

	/* determine the maximum value */
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++) {
			if (array[i][j] > max)
				max = array[i][j];
		}

	/* compute scaling factor */
	factor = 1.0 / max;

	/* white is maximum, black is zero */
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++) {
			image[i][j] = (array[i][j]) * factor * 255;
		}
}

/*----------------------------------------------------------------------------*/

void init_level_set(double **f, /* in+out  : the level set function */
int nx, /* in  : size in x-direction */
int ny, /* in  : size in y-direction */
int bx, /* in  : boundary size in x-direction */
int by, /* in  : boundary size in y-direction */
double hx, /* in  : the grid size in x-dir. */
double hy, /* in  : the grid size in y-dir. */
int init_type /* in  : the type of level set initialisation */
) {
	int i, j, k, l;
	float x, y;
	float rad;

	switch (init_type) {

	case SQUARE:

		/* initialise the zero level set as a square */
		set_matrix_2d(f, nx, ny, bx, by, 1.0);
		set_matrix_boundary(f, nx, ny, bx, by, 11, 11, 0.0);
		set_matrix_boundary(f, nx, ny, bx, by, 10, 10, -1.0);

		break;

	case GRID:

		/* initialise the zero level set as a grid of squares*/
		set_matrix_2d(f, nx, ny, bx, by, -0.5);
		for (i = bx + 5; i < nx + bx - 6; i += 10)
			for (j = by + 5; j < ny + by - 6; j += 10) {

				for (k = i; k < i + 5; k++)
					for (l = j; l < j + 5; l++)
						f[k][l] = 10;
			}

		break;

	case CIRCLE:

		/* initialise the zero level set as a circle */
		set_matrix_2d(f, nx, ny, bx, by, -0.5);
		if (nx < ny)
			rad = 0.95 * nx / 2;
		//rad=0.10*nx/2;
		else
			rad = 0.95 * ny / 2;
		//rad=0.10*ny/2;
		for (i = bx; i < nx + bx; i++)
			for (j = by; j < ny + by; j++) {
				x = (i - bx - nx / 2);
				y = (j - by - ny / 2);

				if (x * x + y * y < rad * rad)
					f[i][j] = 0.5;
				else
					f[i][j] = -0.5;
			}

		break;

	case PRESET:

		/* use the stored solution */
		copy_matrix_2d(phi_stored, f, nx, ny, bx, by);

		break;

	default:

		/* initialise the zero level set as a square */
		set_matrix_2d(f, nx, ny, bx, by, 1.0);
		set_matrix_boundary(f, nx, ny, bx, by, 11, 11, 0.0);
		set_matrix_boundary(f, nx, ny, bx, by, 10, 10, -1.0);
	}
}

/*----------------------------------------------------------------------------*/

void reinit_level_set(double **phi, /* in+out  : the level set function */
int nx, /* in  : size in x-direction */
int ny, /* in  : size in y-direction */
int bx, /* in  : boundary size in x-direction */
int by, /* in  : boundary size in y-direction */
double hx, /* in  : the grid size in x-dir. */
double hy, /* in  : the grid size in y-dir. */
int iter /* in  : number of iterations */
) {
	int i, j, k; /* loop variables */
	double tau; /* time step size */
	double a, b, c, d; /* backward and forward differences in x- and y-direction */
	double a_plus, a_min; /* helper variables */
	double b_plus, b_min; /* helper variables */
	double c_plus, c_min; /* helper variables */
	double d_plus, d_min; /* helper variables */
	double G; /* flux function */
	double **sign; /* the sign of phi */
	double **u; /* the evolving image */
	double **u_new; /* the updated image */
	double **tmp; /* temporary pointer */

	/* allocate memory */
	ALLOC_MATRIX(1, nx + 2 * bx, ny + 2 * by, &u);
	ALLOC_MATRIX(1, nx + 2 * bx, ny + 2 * by, &u_new);
	ALLOC_MATRIX(1, nx + 2 * bx, ny + 2 * by, &sign);

	/* initialisations */
	tau = 0.025;

	/* initialise arrays */
	copy_matrix_2d(phi, u, nx, ny, bx, by);
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++)
			sign[i][j] = signum(phi[i][j]); //phi[i][j]/sqrt(phi[i][j]*phi[i][j] + 0.1*0.1);//

		/* explicit time stepping */
	for (k = 0; k < iter; k++) {

		/* mirror the boundaries */
		mirror_bounds_2d(u, nx, ny, bx, by);

		/* update the signal in all pixels */
		for (i = bx; i < nx + bx; i++)
			for (j = by; j < ny + by; j++) {

				/* compute the differences */
				a = (u[i][j] - u[i - 1][j]) / hx;
				b = (u[i + 1][j] - u[i][j]) / hx;
				c = (u[i][j] - u[i][j - 1]) / hy;
				d = (u[i][j + 1] - u[i][j]) / hy;

				/* compute the flux function */
				if (sign[i][j] > 0.0) {

					a_plus = max(a, 0.0);
					b_min = min(b, 0.0);
					c_plus = max(c, 0.0);
					d_min = min(d, 0.0);
					G = sqrt(
							max(a_plus * a_plus, b_min * b_min)
									+ max(c_plus * c_plus, d_min * d_min))
							- 1.0;
				} else if (sign[i][j] < 0.0) {

					a_min = min(a, 0.0);
					b_plus = max(b, 0.0);
					c_min = min(c, 0.0);
					d_plus = max(d, 0.0);
					G = sqrt(
							max(a_min * a_min, b_plus * b_plus)
									+ max(c_min * c_min, d_plus * d_plus))
							- 1.0;
				} else
					G = 0.0;

				u_new[i][j] = u[i][j] - tau * sign[i][j] * G;
			}

		/* swap pointers */
		tmp = u;
		u = u_new;
		u_new = tmp;
	}

	/* copy solution back */
	copy_matrix_2d(u, phi, nx, ny, bx, by);

	/* free memory */
	FREE_MATRIX(1, nx + 2 * bx, ny + 2 * by, u);
	FREE_MATRIX(1, nx + 2 * bx, ny + 2 * by, u_new);
	FREE_MATRIX(1, nx + 2 * bx, ny + 2 * by, sign);
}

/*----------------------------------------------------------------------------*/

void signed_distance(double **f, /* in+out  : a level set function */
int nx, /* in  : size in x-direction */
int ny, /* in  : size in y-direction */
int bx, /* in  : boundary size in x-direction */
int by /* in  : boundary size in y-direction */
) {
	int i, j; /* loop variables */

	double **phi_help_1; /* helper variable */
	double **phi_help_2; /* helper variable */

	/* allocate memory */
	ALLOC_MATRIX(2, nx + 2 * bx, ny + 2 * by, &phi_help_1, &phi_help_2);

	/* convert the level set function to a binary image such that
	 phi > 0.0 in the foreground
	 phi < 0.0 in the background */
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++) {

			if (f[i][j] < 0.0)
				phi_help_1[i][j] = -BIGNUMBER;
			else
				phi_help_1[i][j] = 0.0;
		}
	/* convert the level set function to a binary image such that
	 phi < 0.0 in the foreground
	 phi > 0.0 in the background */
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++) {

			if (f[i][j] < 0.0)
				phi_help_2[i][j] = 0.0;
			else
				phi_help_2[i][j] = -BIGNUMBER;
		}

	/* fill in the background with the Euclidean Distance Transform */
	EDT(phi_help_1, nx, ny, bx, by);
	EDT(phi_help_2, nx, ny, bx, by);

	/* copy the distances back */
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++) {

			if (f[i][j] < 0.0)
				f[i][j] = -phi_help_1[i][j];
			else
				f[i][j] = phi_help_2[i][j];
		}

	/* free memory */
	FREE_MATRIX(2, nx + 2 * bx, ny + 2 * by, phi_help_1, phi_help_2);
}

/*----------------------------------------------------------------------------*/

void EDT(double **ff, /* in+out  : binary image, foreground = 0.0, background = -inf */
int nx, /* in  : size in x-direction */
int ny, /* in  : size in y-direction */
int bx, /* in  : boundary size in x-direction */
int by /* in  : boundary size in y-direction */
) {
	int i, j, k; /* loop variables */
	double max, help;

	double **u; /* the evolving image */

	/* allocate memory */
	ALLOC_MATRIX(1, nx + 2 * bx, ny + 2 * by, &u);

	/* 1D dilation in i direction */
	for (j = by; j < ny + by; j++) {
		for (i = bx; i < nx + bx; i++) {

			max = ff[i][j];
			for (k = bx; k < nx + bx; k++) {

				help = ff[k][j] - (k - i) * (k - i);
				if (help > max)
					max = help;
			}

			u[i][j] = max;

		}
	}
	copy_matrix_2d(u, ff, nx, ny, bx, by);

	int hh = 0;
	/* 1D dilation in j direction */
	for (i = bx; i < nx + bx; i++) {
		for (j = by; j < ny + by; j++) {

			max = ff[i][j];
			for (k = by; k < ny + by; k++) {
				help = ff[i][k] - ((k - j) * (k - j));
				if (help > max) {
					hh = k;
					max = help;
				}
			}
			u[i][j] = max;
		}
	}

	copy_matrix_2d(u, ff, nx, ny, bx, by);

	/* take the square root */
	for (j = by; j < ny + by; j++)
		for (i = bx; i < nx + bx; i++) {

			ff[i][j] = sqrt(-ff[i][j]);
		}
	/* free memory */
	FREE_MATRIX(1, nx + 2 * bx, ny + 2 * by, u);
}

/*----------------------------------------------------------------------------*/



void CV(double **f, 								/* in  : the image  */
double **phi1, double **phi2, int nx,				/* in  : size in x-direction */
int ny, 											/* in  : size in y-direction */
int bx,												/* in  : boundary size in x-direction */
int by,												/* in  : boundary size in y-direction */
double hx, 											/* in  : the grid size in x-dir. */
double hy,											/* in  : the grid size in y-dir. */
double tau, 										/* in  : time step size */
double T,											/* in  : stopping time */
double mu, 											/* in  : weighting parameter for MCM */
double nu,											/* in  : weighting parameter for the area inside the curve */
double epsilon, 									/* in  : regularisation parameter for the Heaviside function */
double lambda11, double lambda12, double lambda21, double lambda22)

/* Chan-Vese implementation of the Chan-Vese model */
/* An implicit time stepping scheme according to the Chan-Vese paper solved with sor
 on every time step. The value of the parameter p in the original paper is taken to
 be 1, such that the length of the zero level line is not calculated. */

{
	int i, j, k; 									/* loop variables */
	int iter; 										/* the number of sor iterations */
	double t; 										/* time */
	double omega; 									/* overrelaxation parameter */
	double hx2, hy2, hx_2, hy_2; 					/* time savers */
	double a, b; 									/* forward and central finite differences */

	double c_11, c_12, c_21, c_22;
	double old_c_11, old_c_12, old_c_21, old_c_22;
	double area_11, area_12, area_21, area_22;
	double H1, H2;

	int M; 											/* number of pixels in the stopping criterium */
	double Q; 										/* stopping quotient */

	double **phi1_C_1, **phi1_C_2, **phi1_C_3, **phi1_C_4;
	double **phi2_C_1, **phi2_C_2, **phi2_C_3, **phi2_C_4;
	double **delta1;
	double **delta2;
	double **rhs1;
	double **rhs2;
	double **phi_old_1;
	double **phi_old_2;

	double sum1;
	double sum2;
	double sum3;
	double sum4;

	/* for output */
	int counter = 1; 							/* iteration counter */
	char *prefix;								/* the prefix indicating the iteration level */

	/* intialisations */
	Q = tau * hx * hy + 1.0;
	int counterX = 0;
	iter = 5;
	omega = 1;		//0.95;
	double difference_c = 0;


	/* allocate memory */
	ALLOC_MATRIX(4, nx + 2 * bx, ny + 2 * by, &phi1_C_1, &phi1_C_2, &phi1_C_3,&phi1_C_4);
	ALLOC_MATRIX(4, nx + 2 * bx, ny + 2 * by, &phi2_C_1, &phi2_C_2, &phi2_C_3,&phi2_C_4);
	ALLOC_MATRIX(2, nx + 2 * bx, ny + 2 * by, &delta1, &delta2);
	ALLOC_MATRIX(2, nx + 2 * bx, ny + 2 * by, &rhs1, &rhs2);
	ALLOC_MATRIX(2, nx + 2 * bx, ny + 2 * by, &phi_old_1, &phi_old_2);

	/* mirror the boundaries */
	mirror_bounds_2d(f, nx, ny, bx, by);

	/* initialise phi_old */
	copy_matrix_2d(phi1, phi_old_1, nx, ny, bx, by);
	copy_matrix_2d(phi2, phi_old_2, nx, ny, bx, by);

	/* compute time savers */
	hx2 = (2.0 * hx);
	hy2 = (2.0 * hy);
	hx_2 = (hx * hx);
	hy_2 = (hy * hy);


	level_set_to_image(phi1, phi2, nx, ny, bx, by, out);
	if (draw)
		handleDraw();

	c_11 = c_12 = c_21 = c_22 = 0.0;

	/* time stepping */
//    for(t = 0.0; t< 10; t += tau)
	for (t = 0.0; Q > tau * hx * hy; t += tau) { //t < T; t += tau){ //

		/* reinitialise the level set function */
		signed_distance(phi1, nx, ny, bx, by);
		signed_distance(phi2, nx, ny, bx, by);
		/* mirror the boundaries */
		//if(counterX%100==0)
		//	printf("%d \n", counterX);

		mirror_bounds_2d(phi1, nx, ny, bx, by);
		mirror_bounds_2d(phi2, nx, ny, bx, by);
		mirror_bounds_2d(phi_old_1, nx, ny, bx, by);
		mirror_bounds_2d(phi_old_2, nx, ny, bx, by);
//===================================================================================================================================
		/* initialise */

		old_c_11 = c_11;
		old_c_12 = c_12;
		old_c_21 = c_21;
		old_c_22 = c_22;

		c_11 = c_12 = c_21 = c_22 = 0.0;
		area_11 = area_12 = area_21 = area_22 = 0.0;

		/* update the mean values of both regions */
		for (i = bx; i < nx + bx; i++) {
			for (j = by; j < ny + by; j++) {
				/* compute the Heaviside function */
				H1 = heaviside_function(phi_old_1[i][j], epsilon);
				H2 = heaviside_function(phi_old_2[i][j], epsilon);

				c_11 += f[i][j] * H1 * H2;
				area_11 += H1 * H2;

				c_12 += f[i][j] * H1 * (1.0 - H2);
				area_12 += (H1) * (1.0 - H2);

				c_21 += f[i][j] * (1.0 - H1) * H2;
				area_21 += (1.0 - H1) * (H2);

				c_22 += f[i][j] * (1.0 - H1) * (1.0 - H2);
				area_22 += (1.0 - H1) * (1.0 - H2);
			}
		}

		c_11 /= area_11;
		c_12 /= area_12;
		c_21 /= area_21;
		c_22 /= area_22;

		// print out info
		if (counterX % 100 == 0)
		{
			printf("(%4d) c_11: %4.1f\tc_12: %4.1f\tc_21: %4.1f\tc_22: %4.1f\tdiff: %f\n", counterX, c_11, c_12, c_21, c_22, difference_c);
			printf("(    ) a_11: %4.1f\ta_12: %4.1f\ta_21: %4.1f\ta_22: %4.1f\tsum= %.0f\n", area_11, area_12, area_21, area_22, area_11+area_12+area_21+area_22);
			printf("\n");
		}

//=============================================== PHI 1 ====================================================================
		/* compute the coefficients that depend on the values of the previous time step */
		double newVar = tau / hx * hx;
		double newVar2;
		double C_dem;
		for (i = bx; i < nx + bx; i++)
		{
			for (j = by; j < ny + by; j++)
			{
				a = (phi_old_1[i + 1][j] - phi_old_1[i][j]) / hx;
				b = (phi_old_1[i][j + 1] - phi_old_1[i][j - 1]) / hy2;

				if ((a == 0.0) && (b == 0.0))
					phi1_C_1[i][j] = 0.0;
				else
					phi1_C_1[i][j] = 1.0 / sqrt(a * a + b * b);

				a = (phi_old_1[i][j] - phi_old_1[i - 1][j]) / hx;
				b = (phi_old_1[i - 1][j + 1] - phi_old_1[i - 1][j - 1]) / hy2;

				if ((a == 0.0) && (b == 0.0))
					phi1_C_2[i][j] = 0.0;
				else
					phi1_C_2[i][j] = 1.0 / sqrt(a * a + b * b);

				a = (phi_old_1[i][j + 1] - phi_old_1[i][j]) / hy;
				b = (phi_old_1[i + 1][j] - phi_old_1[i - 1][j]) / hx2;

				if ((a == 0.0) && (b == 0.0))
					phi1_C_3[i][j] = 0.0;
				else
					phi1_C_3[i][j] = 1.0 / sqrt(a * a + b * b);

				a = (phi_old_1[i][j] - phi_old_1[i][j - 1]) / hy;
				b = (phi_old_1[i + 1][j - 1] - phi_old_1[i - 1][j - 1]) / hx2;

				if ((a == 0.0) && (b == 0.0))
					phi1_C_4[i][j] = 0.0;
				else
					phi1_C_4[i][j] = 1.0 / sqrt(a * a + b * b);


				delta1[i][j] = delta_function(phi_old_1[i][j], epsilon);

				sum1 = lambda11 * ((f[i][j] - c_11)* (f[i][j] - c_11)) * (H2);
				sum2 = lambda12 * ((f[i][j] - c_12)* (f[i][j] - c_12)) * (1.0 - H2);
				sum3 = lambda21 * ((f[i][j] - c_21)* (f[i][j] - c_21)) * (H2);
				sum4 = lambda22 * ((f[i][j] - c_22)* (f[i][j] - c_22)) * (1.0 - H2);

				rhs1[i][j] = tau * delta1[i][j] * (- sum1 - sum2 + sum3 + sum4);

				C_dem = (1.0+ newVar * delta1[i][j] * mu* (phi1_C_1[i][j] + phi1_C_2[i][j]+ phi1_C_3[i][j] + phi1_C_4[i][j]));

				newVar2 = phi1_C_1[i][j] * phi_old_1[i + 1][j]
						+ phi1_C_2[i][j] * phi_old_1[i - 1][j]
						+ phi1_C_3[i][j] * phi_old_1[i][j + 1]
						+ phi1_C_4[i][j] * phi_old_1[i][j - 1];

				phi1[i][j] = (1.0 - omega) * phi_old_1[i][j]+ ((omega)* (phi_old_1[i][j]+ (newVar * delta1[i][j] * mu * newVar2)+ rhs1[i][j]));
				phi1[i][j] = phi1[i][j] / C_dem;

			}
		}

//		for(k = 0; k < iter; k++)
//		{
//			for(i = bx; i < nx + bx; i++)
//			{
//				for(j = by; j < ny + by; j++)
//				{
//					C_dem = (1.0 + newVar*delta1[i][j]*mu*(phi1_C_1[i][j] +phi1_C_2[i][j] + phi1_C_3[i][j] + phi1_C_4[i][j]));
//					newVar2 = phi1_C_1[i][j]*phi1[i+1][j]
//							+ phi1_C_2[i][j]*phi1[i-1][j]
//							+ phi1_C_3[i][j]*phi1[i][j+1]
//							+ phi1_C_4[i][j]*phi1[i][j-1];
//
//					phi1[i][j] = (1.0-omega) * phi1[i][j] +
//								omega * ((newVar*delta1[i][j]*mu*newVar2) + rhs1[i][j]);
//					phi1[i][j] = phi1[i][j] / C_dem;
//				}
//			}
//		}
//=============================================== PHI 2 ====================================================================
		/* compute the coefficients that depend on the values of the previous time step */
		for (i = bx; i < nx + bx; i++) {
			for (j = by; j < ny + by; j++) {
				a = (phi_old_2[i + 1][j] - phi_old_2[i][j]) / hx;
				b = (phi_old_2[i][j + 1] - phi_old_2[i][j - 1]) / hy2;

				if ((a == 0.0) && (b == 0.0))
					phi2_C_1[i][j] = 0.0;
				else
					phi2_C_1[i][j] = 1.0 / sqrt(a * a + b * b);

				a = (phi_old_2[i][j] - phi_old_2[i - 1][j]) / hx;
				b = (phi_old_2[i - 1][j + 1] - phi_old_2[i - 1][j - 1]) / hy2;

				if ((a == 0.0) && (b == 0.0))
					phi2_C_2[i][j] = 0.0;
				else
					phi2_C_2[i][j] = 1.0 / sqrt(a * a + b * b);

				a = (phi_old_2[i][j + 1] - phi_old_2[i][j]) / hy;
				b = (phi_old_2[i + 1][j] - phi_old_2[i - 1][j]) / hx2;

				if ((a == 0.0) && (b == 0.0))
					phi2_C_3[i][j] = 0.0;
				else
					phi2_C_3[i][j] = 1.0 / sqrt(a * a + b * b);

				a = (phi_old_2[i][j] - phi_old_2[i][j - 1]) / hy;
				b = (phi_old_2[i + 1][j - 1] - phi_old_2[i - 1][j - 1]) / hx2;

				if ((a == 0.0) && (b == 0.0))
					phi2_C_4[i][j] = 0.0;
				else
					phi2_C_4[i][j] = 1.0 / sqrt(a * a + b * b);

				delta2[i][j] = delta_function(phi_old_2[i][j], epsilon);


				sum1 = lambda11 * ((f[i][j] - c_11) * (f[i][j] - c_11)) * (H1);
				sum2 = lambda12 * ((f[i][j] - c_12) * (f[i][j] - c_12)) * (H1);
				sum3 = lambda21 * ((f[i][j] - c_21) * (f[i][j] - c_21)) * (1.0 - H1);
				sum4 = lambda22 * ((f[i][j] - c_22) * (f[i][j] - c_22)) * (1.0 - H1);


				rhs2[i][j] = tau * delta2[i][j] * (- sum1 + sum2 - sum3 + sum4);

				C_dem = (1.0 + newVar * delta2[i][j] * mu *
						(phi2_C_1[i][j] + phi2_C_2[i][j]
					   + phi2_C_3[i][j] + phi2_C_4[i][j]));

				newVar2 = phi2_C_1[i][j] * phi_old_2[i + 1][j]
						+ phi2_C_2[i][j] * phi_old_2[i - 1][j]
						+ phi2_C_3[i][j] * phi_old_2[i][j + 1]
						+ phi2_C_4[i][j] * phi_old_2[i][j - 1];

				phi2[i][j] = (1.0 - omega) * phi_old_2[i][j] + omega * (phi_old_2[i][j] + (newVar * delta2[i][j] * mu * newVar2) + rhs2[i][j]);
				phi2[i][j] = phi2[i][j] / C_dem;
			}
		}

//				for(k = 0; k < iter; k++)
//				{
//					for(i = bx; i < nx + bx; i++)
//					{
//						for(j = by; j < ny + by; j++)
//						{
//							C_dem = (1.0 + newVar*delta2[i][j]*mu*(phi2_C_1[i][j] +phi2_C_2[i][j] + phi2_C_3[i][j] + phi2_C_4[i][j]));
//							newVar2 = phi2_C_1[i][j]*phi2[i+1][j]
//									+ phi2_C_2[i][j]*phi2[i-1][j]
//									+ phi2_C_3[i][j]*phi2[i][j+1]
//									+ phi2_C_4[i][j]*phi2[i][j-1];
//
//							phi2[i][j] = (1.0-omega) * phi2[i][j] +
//										omega * ((newVar*delta2[i][j]*mu*newVar2) + rhs2[i][j]);
//							phi2[i][j] = phi2[i][j] / C_dem;
//
//
//						}
//					}
//				}
//===================================================================================================================================
		/* stopping criterium */
		difference_c = fabs((old_c_11 + old_c_12 + old_c_21 + old_c_22)- (c_11 + c_12 + c_21 + c_22));

//		printf("%d) = %3.3f \n", counterX, difference_c);
//		if(counterX%100==0)
//			printf("%d) = %3.5f \n", counterX, difference_c);

		//END for loop:     for(t = 0.0; Q > tau*hx*hy; t += tau)


		if(boolean_of_colors_hard_coded)
		{
			final_c11 = small_details_gray_level;
			final_c12 = background_gray_level;
			final_c21 = skull_gray_level ;
			final_c22 = tumor_gray_level;
		}
		else
		{
			final_c11 = c_11;
			final_c12 = c_12;
			final_c21 = c_21;
			final_c22 = c_22;
		}
//		printf("%d) Final = %5.2f, %5.2f, %5.2f, %5.2f \n", counterX, final_c11, final_c12, final_c21, final_c22);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		if(regular_evaluation && counterX%regular_evaluation_int==0)
		{
			printf("(%d)--",counterX);
			level_set_to_image(phi1, phi2, nx, ny, bx, by, out);
			post_segmented_image(out, nx, ny, bx, by);
			old_global_f_measure = global_f_measure;
			evaluate(f_post_seg, f_ot, nx, ny, bx, by);
			printf("\n");
		}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		counterX++;
		if (counterX > maximumNoIterations || difference_c < minimumDifference) {

			printf("(%d) c_11: %f, c_12: %f, c_21: %f, c_22: %f, difference: %f\n", counterX, c_11, c_12, c_21, c_22, difference_c);
			printf("Stopping Criteria: \n");
			printf("Iterations No: %d, difference: %f\n", counterX,difference_c);
			Q = 0;
		}

		copy_matrix_2d(phi1, phi_old_1, nx, ny, bx, by);
		copy_matrix_2d(phi2, phi_old_2, nx, ny, bx, by);
//===================================================================================================================================
		/* update visualisation */
		level_set_to_image(phi1, phi2, nx, ny, bx, by, out);
		if (draw)
			handleDraw();
	}

//===================================================================================================================================
	/* free memory */
	FREE_MATRIX(4, nx + 2 * bx, ny + 2 * by, phi1_C_1, phi1_C_2, phi1_C_3,phi1_C_4);
	FREE_MATRIX(4, nx + 2 * bx, ny + 2 * by, phi2_C_1, phi2_C_2, phi2_C_3,phi2_C_4);
	FREE_MATRIX(2, nx + 2 * bx, ny + 2 * by, delta1, delta2);
	FREE_MATRIX(2, nx + 2 * bx, ny + 2 * by, rhs1, rhs2);
	FREE_MATRIX(2, nx + 2 * bx, ny + 2 * by, phi_old_1, phi_old_2);
}

void CV_vec_two_channels(
double **f_c1,
double **f_c2,
double **phi1, double **phi2,
int nx, int ny, int bx, int by,
double hx, double hy, double tau, double T,
double mu, double nu, double epsilon,
double lambda11, double lambda12, double lambda21, double lambda22){

	int i, j, k;
	int iter;
	double t;
	double omega;
	double hx2, hy2, hx_2, hy_2;
	double a, b;

	double c_11_chan_1, c_12_chan_1, c_21_chan_1, c_22_chan_1;
	double c_11_chan_2, c_12_chan_2, c_21_chan_2, c_22_chan_2;

	double old_c_11_chan_1, old_c_12_chan_1, old_c_21_chan_1, old_c_22_chan_1;
	double old_c_11_chan_2, old_c_12_chan_2, old_c_21_chan_2, old_c_22_chan_2;

	double area_11, area_12, area_21, area_22;
	double H1, H2;

	int M;
	double Q;

	double **phi1_C_1, **phi1_C_2, **phi1_C_3, **phi1_C_4;
	double **phi2_C_1, **phi2_C_2, **phi2_C_3, **phi2_C_4;
	double **delta1;
	double **delta2;
	double **rhs1;
	double **rhs2;
	double **phi_old_1;
	double **phi_old_2;

	double sum1;
	double sum2;
	double sum3;
	double sum4;

	/* intialisations */
	Q = tau * hx * hy + 1.0;
	int counterX = 0;
	iter = 5;
	omega = 1;		//0.95;

	/* allocate memory */
	ALLOC_MATRIX(4, nx + 2 * bx, ny + 2 * by, &phi1_C_1, &phi1_C_2, &phi1_C_3,&phi1_C_4);
	ALLOC_MATRIX(4, nx + 2 * bx, ny + 2 * by, &phi2_C_1, &phi2_C_2, &phi2_C_3,&phi2_C_4);
	ALLOC_MATRIX(2, nx + 2 * bx, ny + 2 * by, &delta1, &delta2);
	ALLOC_MATRIX(2, nx + 2 * bx, ny + 2 * by, &rhs1, &rhs2);
	ALLOC_MATRIX(2, nx + 2 * bx, ny + 2 * by, &phi_old_1, &phi_old_2);

	/* mirror the boundaries */
	mirror_bounds_2d(f_c1, nx, ny, bx, by);
	mirror_bounds_2d(f_c2, nx, ny, bx, by);
	mirror_bounds_2d(f_c3, nx, ny, bx, by);

	/* initialise phi_old */
	copy_matrix_2d(phi1, phi_old_1, nx, ny, bx, by);
	copy_matrix_2d(phi2, phi_old_2, nx, ny, bx, by);

	/* compute time savers */
	hx2 = (2.0 * hx);
	hy2 = (2.0 * hy);
	hx_2 = (hx * hx);
	hy_2 = (hy * hy);

	level_set_to_image(phi1, phi2, nx, ny, bx, by, out);
	if (draw)
		handleDraw();

	c_11_chan_1 = c_12_chan_1 = c_21_chan_1 = c_22_chan_1 = 0.0;
	c_11_chan_2 = c_12_chan_2 = c_21_chan_2 = c_22_chan_2 = 0.0;


	/* time stepping */
	for (t = 0.0; Q > tau * hx * hy; t += tau) { //t < T; t += tau){ //

//		printf("\nt: %f, T: %f, mu: %f, nu: %f\n", t, T, mu, nu);

		/* reinitialise the level set function */
		signed_distance(phi1, nx, ny, bx, by);
		signed_distance(phi2, nx, ny, bx, by);

		/* mirror the boundaries */
		mirror_bounds_2d(phi1, nx, ny, bx, by);
		mirror_bounds_2d(phi2, nx, ny, bx, by);
		mirror_bounds_2d(phi_old_1, nx, ny, bx, by);
		mirror_bounds_2d(phi_old_2, nx, ny, bx, by);

		/* initialise */
		old_c_11_chan_1 = c_11_chan_1;
		old_c_12_chan_1 = c_12_chan_1;
		old_c_21_chan_1 = c_21_chan_1;
		old_c_22_chan_1 = c_22_chan_1;

		old_c_11_chan_2 = c_11_chan_2;
		old_c_12_chan_2 = c_12_chan_2;
		old_c_21_chan_2 = c_21_chan_2;
		old_c_22_chan_2 = c_22_chan_2;

		c_11_chan_1 = c_12_chan_1 = c_21_chan_1 = c_22_chan_1 = 0.0;
		c_11_chan_2 = c_12_chan_2 = c_21_chan_2 = c_22_chan_2 = 0.0;

		area_11 = area_12 = area_21 = area_22 = 0.0;

		/* update the mean values of both regions */
		for (i = bx; i < nx + bx; i++)
			for (j = by; j < ny + by; j++) {
				/* compute the Heaviside function */
				H1 = heaviside_function(phi_old_1[i][j], epsilon);
				H2 = heaviside_function(phi_old_2[i][j], epsilon);

				c_11_chan_1 += f_c1[i][j] * H1 * H2;
				c_11_chan_2 += f_c2[i][j] * H1 * H2;
				area_11 += H1 * H2;

				c_12_chan_1 += f_c1[i][j] * H1 * (1.0 - H2);
				c_12_chan_2 += f_c2[i][j] * H1 * (1.0 - H2);
				area_12 += (H1) * (1.0 - H2);

				c_21_chan_1 += f_c1[i][j] * (1.0 - H1) * H2;
				c_21_chan_2 += f_c2[i][j] * (1.0 - H1) * H2;
				area_21 += (1.0 - H1) * (H2);

				c_22_chan_1 += f_c1[i][j] * (1.0 - H1) * (1.0 - H2);
				c_22_chan_2 += f_c2[i][j] * (1.0 - H1) * (1.0 - H2);
				area_22 += (1.0 - H1) * (1.0 - H2);
			}
		c_11_chan_1 /= area_11;
		c_11_chan_2 /= area_11;

		c_12_chan_1 /= area_12;
		c_12_chan_2 /= area_12;

		c_21_chan_1 /= area_21;
		c_21_chan_2 /= area_21;

		c_22_chan_1 /= area_22;
		c_22_chan_2 /= area_22;

//		printf("\n chan1: c_11: %.2f, c_12: %.2f, c_21: %.2f, c_22: %.2f", c_11_chan_1, c_12_chan_1, c_21_chan_1, c_22_chan_1);
//		printf("\n chan2: c_11: %.2f, c_12: %.2f, c_21: %.2f, c_22: %.2f", c_11_chan_2, c_12_chan_2, c_21_chan_2, c_22_chan_2);

//=============================================== PHI 1 ====================================================================
		/* compute the coefficients that depend on the values of the previous time step */
		double newVar = tau / hx * hx;

		double newVar2;
		double C_dem;

		for (i = bx; i < nx + bx; i++)
			for (j = by; j < ny + by; j++) {

				a = (phi_old_1[i + 1][j] - phi_old_1[i][j]) / hx;
				b = (phi_old_1[i][j + 1] - phi_old_1[i][j - 1]) / hy2;
				if ((a == 0.0) && (b == 0.0))
					phi1_C_1[i][j] = 0.0;
				else
					phi1_C_1[i][j] = 1.0 / sqrt(a * a + b * b);

				a = (phi_old_1[i][j] - phi_old_1[i - 1][j]) / hx;
				b = (phi_old_1[i - 1][j + 1] - phi_old_1[i - 1][j - 1]) / hy2;
				if ((a == 0.0) && (b == 0.0))
					phi1_C_2[i][j] = 0.0;
				else
					phi1_C_2[i][j] = 1.0 / sqrt(a * a + b * b);

				a = (phi_old_1[i][j + 1] - phi_old_1[i][j]) / hy;
				b = (phi_old_1[i + 1][j] - phi_old_1[i - 1][j]) / hx2;
				if ((a == 0.0) && (b == 0.0))
					phi1_C_3[i][j] = 0.0;
				else
					phi1_C_3[i][j] = 1.0 / sqrt(a * a + b * b);

				a = (phi_old_1[i][j] - phi_old_1[i][j - 1]) / hy;
				b = (phi_old_1[i + 1][j - 1] - phi_old_1[i - 1][j - 1]) / hx2;
				if ((a == 0.0) && (b == 0.0))
					phi1_C_4[i][j] = 0.0;
				else
					phi1_C_4[i][j] = 1.0 / sqrt(a * a + b * b);


				delta1[i][j] = delta_function(phi_old_1[i][j], epsilon);

				sum1 = ((f_c1[i][j] - c_11_chan_1)* (f_c1[i][j] - c_11_chan_1)) + (f_c2[i][j] - c_11_chan_2)* (f_c2[i][j] - c_11_chan_2);
				sum2 = ((f_c1[i][j] - c_12_chan_1)* (f_c1[i][j] - c_12_chan_1)) + (f_c2[i][j] - c_12_chan_2)* (f_c2[i][j] - c_12_chan_2);
				sum3 = ((f_c1[i][j] - c_21_chan_1)* (f_c1[i][j] - c_21_chan_1)) + (f_c2[i][j] - c_21_chan_2)* (f_c2[i][j] - c_21_chan_2);
				sum4 = ((f_c1[i][j] - c_22_chan_1)* (f_c1[i][j] - c_22_chan_1)) + (f_c2[i][j] - c_22_chan_2)* (f_c2[i][j] - c_22_chan_2);

				sum1 = lambda11 * sum1 * (H2);
				sum2 = lambda12 * sum2 * (1.0 - H2);
				sum3 = lambda21 * sum3 * (H2);
				sum4 = lambda22 * sum4 * (1.0 - H2);

				rhs1[i][j] = tau * delta1[i][j] * (- sum1 - sum2 + sum3 + sum4);

				C_dem = (1.0+ newVar * delta1[i][j] * mu* (phi1_C_1[i][j] + phi1_C_2[i][j] + phi1_C_3[i][j] + phi1_C_4[i][j]));

				newVar2 = phi1_C_1[i][j] * phi_old_1[i + 1][j]
						+ phi1_C_2[i][j] * phi_old_1[i - 1][j]
						+ phi1_C_3[i][j] * phi_old_1[i][j + 1]
						+ phi1_C_4[i][j] * phi_old_1[i][j - 1];

				phi1[i][j] = (1.0 - omega) * phi1[i][j]+ ((omega)* (phi_old_1[i][j]+ (newVar * delta1[i][j] * mu * newVar2)+ rhs1[i][j]));
				phi1[i][j] = phi1[i][j] / C_dem;

			}

//=============================================== PHI 2 ====================================================================
		for (i = bx; i < nx + bx; i++)
			for (j = by; j < ny + by; j++) {

				a = (phi_old_2[i + 1][j] - phi_old_2[i][j]) / hx;
				b = (phi_old_2[i][j + 1] - phi_old_2[i][j - 1]) / hy2;
				if ((a == 0.0) && (b == 0.0))
					phi2_C_1[i][j] = 0.0;
				else
					phi2_C_1[i][j] = 1.0 / sqrt(a * a + b * b);

				a = (phi_old_2[i][j] - phi_old_2[i - 1][j]) / hx;
				b = (phi_old_2[i - 1][j + 1] - phi_old_2[i - 1][j - 1]) / hy2;
				if ((a == 0.0) && (b == 0.0))
					phi2_C_2[i][j] = 0.0;
				else
					phi2_C_2[i][j] = 1.0 / sqrt(a * a + b * b);

				a = (phi_old_2[i][j + 1] - phi_old_2[i][j]) / hy;
				b = (phi_old_2[i + 1][j] - phi_old_2[i - 1][j]) / hx2;
				if ((a == 0.0) && (b == 0.0))
					phi2_C_3[i][j] = 0.0;
				else
					phi2_C_3[i][j] = 1.0 / sqrt(a * a + b * b);

				a = (phi_old_2[i][j] - phi_old_2[i][j - 1]) / hy;
				b = (phi_old_2[i + 1][j - 1] - phi_old_2[i - 1][j - 1]) / hx2;
				if ((a == 0.0) && (b == 0.0))
					phi2_C_4[i][j] = 0.0;
				else
					phi2_C_4[i][j] = 1.0 / sqrt(a * a + b * b);

				delta2[i][j] = delta_function(phi_old_2[i][j], epsilon);

				sum1 = ((f_c1[i][j] - c_11_chan_1)* (f_c1[i][j] - c_11_chan_1)) + (f_c2[i][j] - c_11_chan_2)* (f_c2[i][j] - c_11_chan_2);
				sum2 = ((f_c1[i][j] - c_12_chan_1)* (f_c1[i][j] - c_12_chan_1)) + (f_c2[i][j] - c_12_chan_2)* (f_c2[i][j] - c_12_chan_2);
				sum3 = ((f_c1[i][j] - c_21_chan_1)* (f_c1[i][j] - c_21_chan_1)) + (f_c2[i][j] - c_21_chan_2)* (f_c2[i][j] - c_21_chan_2);
				sum4 = ((f_c1[i][j] - c_22_chan_1)* (f_c1[i][j] - c_22_chan_1)) + (f_c2[i][j] - c_22_chan_2)* (f_c2[i][j] - c_22_chan_2);

				sum1 = lambda11 * sum1 * (H1);
				sum2 = lambda12 * sum2 * (H1);
				sum3 = lambda21 * sum3 * (1.0 - H1);
				sum4 = lambda22 * sum4 * (1.0 - H1);

				rhs2[i][j] = tau * delta2[i][j] * (- sum1 + sum2 - sum3 + sum4);

				C_dem = (1.0 + newVar * delta2[i][j] * mu *
						(phi2_C_1[i][j] + phi2_C_2[i][j]
					   + phi2_C_3[i][j] + phi2_C_4[i][j]));

				newVar2 = phi2_C_1[i][j] * phi_old_2[i + 1][j]
						+ phi2_C_2[i][j] * phi_old_2[i - 1][j]
						+ phi2_C_3[i][j] * phi_old_2[i][j + 1]
						+ phi2_C_4[i][j] * phi_old_2[i][j - 1];

				phi2[i][j] = (1.0 - omega) * phi_old_2[i][j] + omega * (phi_old_2[i][j] + (newVar * delta2[i][j] * mu * newVar2) + rhs2[i][j]);
				phi2[i][j] = phi2[i][j] / C_dem;
			}
//==========================================================================================================================
		/* stopping criterium */

		double difference_c_chan_1 = fabs((old_c_11_chan_1 + old_c_12_chan_1 + old_c_21_chan_1+ old_c_22_chan_1)
									- (c_11_chan_1 + c_12_chan_1 + c_21_chan_1+ c_22_chan_1));
		double difference_c_chan_2 =fabs((old_c_11_chan_2 + old_c_12_chan_2 + old_c_21_chan_2+ old_c_22_chan_2)
								- (c_11_chan_2 + c_12_chan_2 + c_21_chan_2+ c_22_chan_2));
		double difference_c_sum = (difference_c_chan_1 + difference_c_chan_2) / 2;


		// END for loop:     for(t = 0.0; Q > tau*hx*hy; t += tau)

//		final_c11 = 0;
//		final_c12 = 10;
//		final_c21 = 110;
//		final_c22 = 255;


		if(boolean_of_colors_hard_coded)
		{
			final_c11 = small_details_gray_level;
			final_c12 = background_gray_level;
			final_c21 = skull_gray_level ;
			final_c22 = tumor_gray_level;
		}
		else
		{
			final_c11 = (c_11_chan_1 + c_11_chan_2) / 2;
			final_c12 = (c_12_chan_1 + c_12_chan_2) / 2;
			final_c21 = (c_21_chan_1 + c_21_chan_2) / 2;
			final_c22 = (c_22_chan_1 + c_22_chan_2) / 2;
		}




		if (counterX % 100 == 0) {
			printf("%5d) Final = %.2f, %.2f, %.2f, %.2f, diff: %.5f \n",
					counterX, final_c11, final_c12, final_c21, final_c22,difference_c_sum);
		}

		if(regular_evaluation && counterX%regular_evaluation_int==0)
		{
			printf("(%d)--",counterX);
			level_set_to_image(phi1, phi2, nx, ny, bx, by, out);
			post_segmented_image(out, nx, ny, bx, by);
			old_global_f_measure = global_f_measure;
			evaluate(f_post_seg, f_ot, nx, ny, bx, by);
			printf("\n");
		}

		counterX++;
		if (counterX > maximumNoIterations	|| difference_c_sum < minimumDifference) {
			Q = 0;
			printf("Iterations No: %d, difference: %f\n", counterX,difference_c_sum);
		}

		copy_matrix_2d(phi1, phi_old_1, nx, ny, bx, by);
		copy_matrix_2d(phi2, phi_old_2, nx, ny, bx, by);

		/* update visualisation */
		level_set_to_image(phi1, phi2, nx, ny, bx, by, out);
//		if(counterX%100==0)
//		{
//			post_segmented_image(out, nx, ny, bx, by);
//			evaluate(f_post_seg, f_ot, nx, ny, bx, by);
//		}
		if (draw)
			handleDraw();
	}
//===================================================================================================================================
	/* free memory */
	FREE_MATRIX(4, nx + 2 * bx, ny + 2 * by, phi1_C_1, phi1_C_2, phi1_C_3,phi1_C_4);
	FREE_MATRIX(4, nx + 2 * bx, ny + 2 * by, phi2_C_1, phi2_C_2, phi2_C_3,phi2_C_4);
	FREE_MATRIX(2, nx + 2 * bx, ny + 2 * by, delta1, delta2);
	FREE_MATRIX(2, nx + 2 * bx, ny + 2 * by, rhs1, rhs2);
	FREE_MATRIX(2, nx + 2 * bx, ny + 2 * by, phi_old_1, phi_old_2);

};

void CV_vec_three_channels
(double **f_c1, /* in  : the image channel 1 */
double **f_c2, /* in  : the image channel 2 */
double **f_c3, /* in  : the image channel 3 */
double **phi1, double **phi2, int nx, /* in  : size in x-direction */
int ny, /* in  : size in y-direction */
int bx, /* in  : boundary size in x-direction */
int by, /* in  : boundary size in y-direction */
double hx, /* in  : the grid size in x-dir. */
double hy, /* in  : the grid size in y-dir. */
double tau, /* in  : time step size */
double T, /* in  : stopping time */
double mu, /* in  : weighting parameter for MCM */
double nu, /* in  : weighting parameter for the area inside the curve */
double epsilon, /* in  : regularisation parameter for the Heaviside function */
double lambda11, double lambda12, double lambda21, double lambda22)
/* Chan-Vese implementation of the Chan-Vese model for vector-valued images */
/* An implicit time stepping scheme according to the Chan-Vese paper solved with sor
 on every time step. The value of the parameter p in the original paper is taken to
 be 1, such that the length of the zero level line is not calculated. */
{
	int i, j, k; 						// loop variables
	int iter; 							// the number of sor iterations */
	double t; 							// time
	double omega; 						// overrelaxation parameter
	double hx2, hy2, hx_2, hy_2; 		// time savers
	double a, b; 						// forward and central finite differences


	double c_11_chan_1, c_12_chan_1, c_21_chan_1, c_22_chan_1;
	double c_11_chan_2, c_12_chan_2, c_21_chan_2, c_22_chan_2;
	double c_11_chan_3, c_12_chan_3, c_21_chan_3, c_22_chan_3;

	double old_c_11_chan_1, old_c_12_chan_1, old_c_21_chan_1, old_c_22_chan_1;
	double old_c_11_chan_2, old_c_12_chan_2, old_c_21_chan_2, old_c_22_chan_2;
	double old_c_11_chan_3, old_c_12_chan_3, old_c_21_chan_3, old_c_22_chan_3;

	double area_11, area_12, area_21, area_22;
	double H1, H2;

	int M;
	double Q;

	double **phi1_C_1, **phi1_C_2, **phi1_C_3, **phi1_C_4;
	double **phi2_C_1, **phi2_C_2, **phi2_C_3, **phi2_C_4;
	double **delta1;
	double **delta2;
	double **rhs1;
	double **rhs2;
	double **phi_old_1;
	double **phi_old_2;

	double sum1;
	double sum2;
	double sum3;
	double sum4;

	/* intialisations */
	Q = tau * hx * hy + 1.0;
	int counterX = 0;
	iter = 5;
	omega = 1;		//0.95;

	/* allocate memory */
	ALLOC_MATRIX(4, nx + 2 * bx, ny + 2 * by, &phi1_C_1, &phi1_C_2, &phi1_C_3,&phi1_C_4);
	ALLOC_MATRIX(4, nx + 2 * bx, ny + 2 * by, &phi2_C_1, &phi2_C_2, &phi2_C_3,&phi2_C_4);
	ALLOC_MATRIX(2, nx + 2 * bx, ny + 2 * by, &delta1, &delta2);
	ALLOC_MATRIX(2, nx + 2 * bx, ny + 2 * by, &rhs1, &rhs2);
	ALLOC_MATRIX(2, nx + 2 * bx, ny + 2 * by, &phi_old_1, &phi_old_2);

	/* mirror the boundaries */
	mirror_bounds_2d(f_c1, nx, ny, bx, by);
	mirror_bounds_2d(f_c2, nx, ny, bx, by);
	mirror_bounds_2d(f_c3, nx, ny, bx, by);

	/* initialise phi_old */
	copy_matrix_2d(phi1, phi_old_1, nx, ny, bx, by);
	copy_matrix_2d(phi2, phi_old_2, nx, ny, bx, by);

	/* compute time savers */
	hx2 = (2.0 * hx);
	hy2 = (2.0 * hy);
	hx_2 = (hx * hx);
	hy_2 = (hy * hy);

	level_set_to_image(phi1, phi2, nx, ny, bx, by, out);
	if (draw)
		handleDraw();

	c_11_chan_1 = c_12_chan_1 = c_21_chan_1 = c_22_chan_1 = 0.0;
	c_11_chan_2 = c_12_chan_2 = c_21_chan_2 = c_22_chan_2 = 0.0;
	c_11_chan_3 = c_12_chan_3 = c_21_chan_3 = c_22_chan_3 = 0.0;


	/* time stepping */
	for (t = 0.0; Q > tau * hx * hy; t += tau) { //t < T; t += tau){ //

//		printf("\nt: %f, T: %f, mu: %f, nu: %f\n", t, T, mu, nu);

		/* reinitialise the level set function */
		signed_distance(phi1, nx, ny, bx, by);
		signed_distance(phi2, nx, ny, bx, by);

		/* mirror the boundaries */
		mirror_bounds_2d(phi1, nx, ny, bx, by);
		mirror_bounds_2d(phi2, nx, ny, bx, by);
		mirror_bounds_2d(phi_old_1, nx, ny, bx, by);
		mirror_bounds_2d(phi_old_2, nx, ny, bx, by);

		/* initialise */
		old_c_11_chan_1 = c_11_chan_1;
		old_c_12_chan_1 = c_12_chan_1;
		old_c_21_chan_1 = c_21_chan_1;
		old_c_22_chan_1 = c_22_chan_1;

		old_c_11_chan_2 = c_11_chan_2;
		old_c_12_chan_2 = c_12_chan_2;
		old_c_21_chan_2 = c_21_chan_2;
		old_c_22_chan_2 = c_22_chan_2;

		old_c_11_chan_3 = c_11_chan_3;
		old_c_12_chan_3 = c_12_chan_3;
		old_c_21_chan_3 = c_21_chan_3;
		old_c_22_chan_3 = c_22_chan_3;

		c_11_chan_1 = c_12_chan_1 = c_21_chan_1 = c_22_chan_1 = 0.0;
		c_11_chan_2 = c_12_chan_2 = c_21_chan_2 = c_22_chan_2 = 0.0;
		c_11_chan_3 = c_12_chan_3 = c_21_chan_3 = c_22_chan_3 = 0.0;
		area_11 = area_12 = area_21 = area_22 = 0.0;

		/* update the mean values of both regions */
		for (i = bx; i < nx + bx; i++)
			for (j = by; j < ny + by; j++) {
				/* compute the Heaviside function */
				H1 = heaviside_function(phi_old_1[i][j], epsilon);
				H2 = heaviside_function(phi_old_2[i][j], epsilon);

				c_11_chan_1 += f_c1[i][j] * H1 * H2;
				c_11_chan_2 += f_c2[i][j] * H1 * H2;
				c_11_chan_3 += f_c3[i][j] * H1 * H2;
				area_11 += H1 * H2;

				c_12_chan_1 += f_c1[i][j] * H1 * (1.0 - H2);
				c_12_chan_2 += f_c2[i][j] * H1 * (1.0 - H2);
				c_12_chan_3 += f_c3[i][j] * H1 * (1.0 - H2);
				area_12 += (H1) * (1.0 - H2);

				c_21_chan_1 += f_c1[i][j] * (1.0 - H1) * H2;
				c_21_chan_2 += f_c2[i][j] * (1.0 - H1) * H2;
				c_21_chan_3 += f_c3[i][j] * (1.0 - H1) * H2;
				area_21 += (1.0 - H1) * (H2);

				c_22_chan_1 += f_c1[i][j] * (1.0 - H1) * (1.0 - H2);
				c_22_chan_2 += f_c2[i][j] * (1.0 - H1) * (1.0 - H2);
				c_22_chan_3 += f_c3[i][j] * (1.0 - H1) * (1.0 - H2);
				area_22 += (1.0 - H1) * (1.0 - H2);
			}
		c_11_chan_1 /= area_11;
		c_11_chan_2 /= area_11;
		c_11_chan_3 /= area_11;

		c_12_chan_1 /= area_12;
		c_12_chan_2 /= area_12;
		c_12_chan_3 /= area_12;

		c_21_chan_1 /= area_21;
		c_21_chan_2 /= area_21;
		c_21_chan_3 /= area_21;

		c_22_chan_1 /= area_22;
		c_22_chan_2 /= area_22;
		c_22_chan_3 /= area_22;


//		printf("\n chan1: c_11: %.2f, c_12: %.2f, c_21: %.2f, c_22: %.2f", c_11_chan_1, c_12_chan_1, c_21_chan_1, c_22_chan_1);
//		printf("\n chan2: c_11: %.2f, c_12: %.2f, c_21: %.2f, c_22: %.2f", c_11_chan_2, c_12_chan_2, c_21_chan_2, c_22_chan_2);
//		printf("\n chan3: c_11: %.2f, c_12: %.2f, c_21: %.2f, c_22: %.2f\n", c_11_chan_3, c_12_chan_3, c_21_chan_3, c_22_chan_3);

//=============================================== PHI 1 ====================================================================
		/* compute the coefficients that depend on the values of the previous time step */
		double newVar = tau / hx * hx;

		double newVar2;
		double C_dem;

		for (i = bx; i < nx + bx; i++)
			for (j = by; j < ny + by; j++) {

				a = (phi_old_1[i + 1][j] - phi_old_1[i][j]) / hx;
				b = (phi_old_1[i][j + 1] - phi_old_1[i][j - 1]) / hy2;
				if ((a == 0.0) && (b == 0.0))
					phi1_C_1[i][j] = 0.0;
				else
					phi1_C_1[i][j] = 1.0 / sqrt(a * a + b * b);

				a = (phi_old_1[i][j] - phi_old_1[i - 1][j]) / hx;
				b = (phi_old_1[i - 1][j + 1] - phi_old_1[i - 1][j - 1]) / hy2;
				if ((a == 0.0) && (b == 0.0))
					phi1_C_2[i][j] = 0.0;
				else
					phi1_C_2[i][j] = 1.0 / sqrt(a * a + b * b);

				a = (phi_old_1[i][j + 1] - phi_old_1[i][j]) / hy;
				b = (phi_old_1[i + 1][j] - phi_old_1[i - 1][j]) / hx2;
				if ((a == 0.0) && (b == 0.0))
					phi1_C_3[i][j] = 0.0;
				else
					phi1_C_3[i][j] = 1.0 / sqrt(a * a + b * b);

				a = (phi_old_1[i][j] - phi_old_1[i][j - 1]) / hy;
				b = (phi_old_1[i + 1][j - 1] - phi_old_1[i - 1][j - 1]) / hx2;
				if ((a == 0.0) && (b == 0.0))
					phi1_C_4[i][j] = 0.0;
				else
					phi1_C_4[i][j] = 1.0 / sqrt(a * a + b * b);


				delta1[i][j] = delta_function(phi_old_1[i][j], epsilon);

				sum1 = ((f_c1[i][j] - c_11_chan_1)* (f_c1[i][j] - c_11_chan_1)) + (f_c2[i][j] - c_11_chan_2)* (f_c2[i][j] - c_11_chan_2) +(f_c3[i][j] - c_11_chan_3)* (f_c3[i][j] - c_11_chan_3);
				sum2 = ((f_c1[i][j] - c_12_chan_1)* (f_c1[i][j] - c_12_chan_1)) + (f_c2[i][j] - c_12_chan_2)* (f_c2[i][j] - c_12_chan_2) +(f_c3[i][j] - c_12_chan_3)* (f_c3[i][j] - c_12_chan_3);
				sum3 = ((f_c1[i][j] - c_21_chan_1)* (f_c1[i][j] - c_21_chan_1)) + (f_c2[i][j] - c_21_chan_2)* (f_c2[i][j] - c_21_chan_2) +(f_c3[i][j] - c_21_chan_3)* (f_c3[i][j] - c_21_chan_3);
				sum4 = ((f_c1[i][j] - c_22_chan_1)* (f_c1[i][j] - c_22_chan_1)) + (f_c2[i][j] - c_22_chan_2)* (f_c2[i][j] - c_22_chan_2) +(f_c3[i][j] - c_22_chan_3)* (f_c3[i][j] - c_22_chan_3);

				sum1 = lambda11 * sum1 * (H2);
				sum2 = lambda12 * sum2 * (1.0 - H2);
				sum3 = lambda21 * sum3 * (H2);
				sum4 = lambda22 * sum4 * (1.0 - H2);

				rhs1[i][j] = tau * delta1[i][j] * (- sum1 - sum2 + sum3 + sum4);

				C_dem = (1.0+ newVar * delta1[i][j] * mu* (phi1_C_1[i][j] + phi1_C_2[i][j] + phi1_C_3[i][j] + phi1_C_4[i][j]));

				newVar2 = phi1_C_1[i][j] * phi_old_1[i + 1][j]
						+ phi1_C_2[i][j] * phi_old_1[i - 1][j]
						+ phi1_C_3[i][j] * phi_old_1[i][j + 1]
						+ phi1_C_4[i][j] * phi_old_1[i][j - 1];

				phi1[i][j] = (1.0 - omega) * phi1[i][j]+ ((omega)* (phi_old_1[i][j]+ (newVar * delta1[i][j] * mu * newVar2)+ rhs1[i][j]));
				phi1[i][j] = phi1[i][j] / C_dem;

			}

//=============================================== PHI 2 ====================================================================
		for (i = bx; i < nx + bx; i++)
			for (j = by; j < ny + by; j++) {

				a = (phi_old_2[i + 1][j] - phi_old_2[i][j]) / hx;
				b = (phi_old_2[i][j + 1] - phi_old_2[i][j - 1]) / hy2;
				if ((a == 0.0) && (b == 0.0))
					phi2_C_1[i][j] = 0.0;
				else
					phi2_C_1[i][j] = 1.0 / sqrt(a * a + b * b);

				a = (phi_old_2[i][j] - phi_old_2[i - 1][j]) / hx;
				b = (phi_old_2[i - 1][j + 1] - phi_old_2[i - 1][j - 1]) / hy2;
				if ((a == 0.0) && (b == 0.0))
					phi2_C_2[i][j] = 0.0;
				else
					phi2_C_2[i][j] = 1.0 / sqrt(a * a + b * b);

				a = (phi_old_2[i][j + 1] - phi_old_2[i][j]) / hy;
				b = (phi_old_2[i + 1][j] - phi_old_2[i - 1][j]) / hx2;
				if ((a == 0.0) && (b == 0.0))
					phi2_C_3[i][j] = 0.0;
				else
					phi2_C_3[i][j] = 1.0 / sqrt(a * a + b * b);

				a = (phi_old_2[i][j] - phi_old_2[i][j - 1]) / hy;
				b = (phi_old_2[i + 1][j - 1] - phi_old_2[i - 1][j - 1]) / hx2;
				if ((a == 0.0) && (b == 0.0))
					phi2_C_4[i][j] = 0.0;
				else
					phi2_C_4[i][j] = 1.0 / sqrt(a * a + b * b);


				delta2[i][j] = delta_function(phi_old_2[i][j], epsilon);

				sum1 = ((f_c1[i][j] - c_11_chan_1)* (f_c1[i][j] - c_11_chan_1)) + (f_c2[i][j] - c_11_chan_2)* (f_c2[i][j] - c_11_chan_2) +(f_c3[i][j] - c_11_chan_3)* (f_c3[i][j] - c_11_chan_3);
				sum2 = ((f_c1[i][j] - c_12_chan_1)* (f_c1[i][j] - c_12_chan_1)) + (f_c2[i][j] - c_12_chan_2)* (f_c2[i][j] - c_12_chan_2) +(f_c3[i][j] - c_12_chan_3)* (f_c3[i][j] - c_12_chan_3);
				sum3 = ((f_c1[i][j] - c_21_chan_1)* (f_c1[i][j] - c_21_chan_1)) + (f_c2[i][j] - c_21_chan_2)* (f_c2[i][j] - c_21_chan_2) +(f_c3[i][j] - c_21_chan_3)* (f_c3[i][j] - c_21_chan_3);
				sum4 = ((f_c1[i][j] - c_22_chan_1)* (f_c1[i][j] - c_22_chan_1)) + (f_c2[i][j] - c_22_chan_2)* (f_c2[i][j] - c_22_chan_2) +(f_c3[i][j] - c_22_chan_3)* (f_c3[i][j] - c_22_chan_3);

				sum1 = lambda11 * sum1 * (H1);
				sum2 = lambda12 * sum2 * (H1);
				sum3 = lambda21 * sum3 * (1.0 - H1);
				sum4 = lambda22 * sum4 * (1.0 - H1);


				rhs2[i][j] = tau * delta2[i][j] * (- sum1 + sum2 - sum3 + sum4);

				C_dem = (1.0 + newVar * delta2[i][j] * mu *
						(phi2_C_1[i][j] + phi2_C_2[i][j]
					   + phi2_C_3[i][j] + phi2_C_4[i][j]));

				newVar2 = phi2_C_1[i][j] * phi_old_2[i + 1][j]
						+ phi2_C_2[i][j] * phi_old_2[i - 1][j]
						+ phi2_C_3[i][j] * phi_old_2[i][j + 1]
						+ phi2_C_4[i][j] * phi_old_2[i][j - 1];

				phi2[i][j] = (1.0 - omega) * phi_old_2[i][j] + omega * (phi_old_2[i][j] + (newVar * delta2[i][j] * mu * newVar2) + rhs2[i][j]);
				phi2[i][j] = phi2[i][j] / C_dem;


//				rhs2[i][j] = phi_old_2[i][j]+ tau * delta2[i][j]
//						* (-nu
//						+ (-1) * (H1)* ((lambda11)* (f_c1[i][j]- c_11_chan_1)* (f_c1[i][j]- c_11_chan_1)
//									  + (lambda12)* (f_c2[i][j]- c_11_chan_2)* (f_c2[i][j]- c_11_chan_2)
//									  + (lambda21)* (f_c3[i][j]- c_11_chan_3)* (f_c3[i][j]- c_11_chan_3)
//									  + (lambda22)* (f_c4[i][j]- c_11_chan_4)* (f_c4[i][j]- c_11_chan_4))
//						+ (-1) * (-H1)* ((lambda11)* (f_c1[i][j]- c_12_chan_1)* (f_c1[i][j]- c_12_chan_1)
//									   + (lambda12)* (f_c2[i][j]- c_12_chan_2)* (f_c2[i][j]- c_12_chan_2)
//									   + (lambda21)* (f_c3[i][j]- c_12_chan_3)* (f_c3[i][j]- c_12_chan_3)
//									   + (lambda22)* (f_c4[i][j]- c_12_chan_4)* (f_c4[i][j]- c_12_chan_4))
//						+ (-1) * (1.0 - H1)* ((lambda11)* (f_c1[i][j]- c_21_chan_1)* (f_c1[i][j]- c_21_chan_1)
//											+ (lambda12)* (f_c2[i][j]- c_21_chan_2)* (f_c2[i][j]- c_21_chan_2)
//											+ (lambda21)* (f_c3[i][j]- c_21_chan_3)* (f_c3[i][j]- c_21_chan_3)
//											+ (lambda22)* (f_c4[i][j]- c_21_chan_4)* (f_c4[i][j]- c_21_chan_4))
//						+ (-1) * (-1.0 + H1)*((lambda11)* (f_c1[i][j]- c_22_chan_1)* (f_c1[i][j]- c_22_chan_1)
//											+ (lambda12)* (f_c2[i][j]- c_22_chan_2)* (f_c2[i][j]- c_22_chan_2)
//											+ (lambda21)* (f_c3[i][j]- c_22_chan_3)* (f_c3[i][j]- c_22_chan_3)
//											+ (lambda22)* (f_c4[i][j]- c_22_chan_4)* (f_c4[i][j]- c_22_chan_4)));


			}
//==========================================================================================================================
		/* stopping criterium */

		double difference_c_chan_1 = fabs((old_c_11_chan_1 + old_c_12_chan_1 + old_c_21_chan_1+ old_c_22_chan_1)
									- (c_11_chan_1 + c_12_chan_1 + c_21_chan_1+ c_22_chan_1));
		double difference_c_chan_2 =fabs((old_c_11_chan_2 + old_c_12_chan_2 + old_c_21_chan_2+ old_c_22_chan_2)
								- (c_11_chan_2 + c_12_chan_2 + c_21_chan_2+ c_22_chan_2));
		double difference_c_chan_3 =fabs((old_c_11_chan_3 + old_c_12_chan_3 + old_c_21_chan_3+ old_c_22_chan_3)
								- (c_11_chan_3 + c_12_chan_3 + c_21_chan_3+ c_22_chan_3));
		double difference_c_sum = (difference_c_chan_1 + difference_c_chan_2+ difference_c_chan_3) / 3;



		// END for loop:     for(t = 0.0; Q > tau*hx*hy; t += tau)

//		final_c11 = 0;
//		final_c12 = 10;
//		final_c21 = 110;
//		final_c22 = 255;


		if(boolean_of_colors_hard_coded)
		{
			final_c11 = small_details_gray_level;
			final_c12 = background_gray_level;
			final_c21 = skull_gray_level ;
			final_c22 = tumor_gray_level;
		}
		else
		{
			final_c11 = (c_11_chan_1 + c_11_chan_2 + c_11_chan_3 ) / 3;
			final_c12 = (c_12_chan_1 + c_12_chan_2 + c_12_chan_3 ) / 3;
			final_c21 = (c_21_chan_1 + c_21_chan_2 + c_21_chan_3 ) / 3;
			final_c22 = (c_22_chan_1 + c_22_chan_2 + c_22_chan_3 ) / 3;
		}


		if (counterX % 100 == 0) {
			printf("%5d) Final = %.2f, %.2f, %.2f, %.2f, diff: %.5f \n",
					counterX, final_c11, final_c12, final_c21, final_c22,difference_c_sum);
		}

		if(regular_evaluation && counterX%regular_evaluation_int==0)
		{
			printf("(%d)--",counterX);
			level_set_to_image(phi1, phi2, nx, ny, bx, by, out);
			post_segmented_image(out, nx, ny, bx, by);
			old_global_f_measure = global_f_measure;
			evaluate(f_post_seg, f_ot, nx, ny, bx, by);
			printf("\n");
		}

		counterX++;
		if (counterX > maximumNoIterations	|| difference_c_sum < minimumDifference) {
			Q = 0;
			printf("Iterations No: %d, difference: %f\n", counterX,difference_c_sum);
		}

		copy_matrix_2d(phi1, phi_old_1, nx, ny, bx, by);
		copy_matrix_2d(phi2, phi_old_2, nx, ny, bx, by);

		/* update visualisation */
		level_set_to_image(phi1, phi2, nx, ny, bx, by, out);
//		if(counterX%100==0)
//		{
//			post_segmented_image(out, nx, ny, bx, by);
//			evaluate(f_post_seg, f_ot, nx, ny, bx, by);
//		}
		if (draw)
			handleDraw();
	}
//===================================================================================================================================
	/* free memory */
	FREE_MATRIX(4, nx + 2 * bx, ny + 2 * by, phi1_C_1, phi1_C_2, phi1_C_3,phi1_C_4);
	FREE_MATRIX(4, nx + 2 * bx, ny + 2 * by, phi2_C_1, phi2_C_2, phi2_C_3,phi2_C_4);
	FREE_MATRIX(2, nx + 2 * bx, ny + 2 * by, delta1, delta2);
	FREE_MATRIX(2, nx + 2 * bx, ny + 2 * by, rhs1, rhs2);
	FREE_MATRIX(2, nx + 2 * bx, ny + 2 * by, phi_old_1, phi_old_2);
}

void CV_vec_four_channels(
double **f_c1,
double **f_c2,
double **f_c3,
double **f_c4,
double **phi1, double **phi2,
int nx, int ny, int bx, int by,
double hx, double hy, double tau, double T,
double mu, double nu, double epsilon,
double lambda11, double lambda12, double lambda21, double lambda22){


	int i, j, k;
	int iter;
	double t;
	double omega;
	double hx2, hy2, hx_2, hy_2;
	double a, b;

	double c_11_chan_1, c_12_chan_1, c_21_chan_1, c_22_chan_1;
	double c_11_chan_2, c_12_chan_2, c_21_chan_2, c_22_chan_2;
	double c_11_chan_3, c_12_chan_3, c_21_chan_3, c_22_chan_3;
	double c_11_chan_4, c_12_chan_4, c_21_chan_4, c_22_chan_4;

	double old_c_11_chan_1, old_c_12_chan_1, old_c_21_chan_1, old_c_22_chan_1;
	double old_c_11_chan_2, old_c_12_chan_2, old_c_21_chan_2, old_c_22_chan_2;
	double old_c_11_chan_3, old_c_12_chan_3, old_c_21_chan_3, old_c_22_chan_3;
	double old_c_11_chan_4, old_c_12_chan_4, old_c_21_chan_4, old_c_22_chan_4;

	double area_11, area_12, area_21, area_22;
	double H1, H2;

	int M;
	double Q;

	double **phi1_C_1, **phi1_C_2, **phi1_C_3, **phi1_C_4;
	double **phi2_C_1, **phi2_C_2, **phi2_C_3, **phi2_C_4;
	double **delta1;
	double **delta2;
	double **rhs1;
	double **rhs2;
	double **phi_old_1;
	double **phi_old_2;

	double sum1;
	double sum2;
	double sum3;
	double sum4;

	/* intialisations */
	Q = tau * hx * hy + 1.0;
	int counterX = 0;
	iter = 5;
	omega = 1;		//0.95;

	/* allocate memory */
	ALLOC_MATRIX(4, nx + 2 * bx, ny + 2 * by, &phi1_C_1, &phi1_C_2, &phi1_C_3,&phi1_C_4);
	ALLOC_MATRIX(4, nx + 2 * bx, ny + 2 * by, &phi2_C_1, &phi2_C_2, &phi2_C_3,&phi2_C_4);
	ALLOC_MATRIX(2, nx + 2 * bx, ny + 2 * by, &delta1, &delta2);
	ALLOC_MATRIX(2, nx + 2 * bx, ny + 2 * by, &rhs1, &rhs2);
	ALLOC_MATRIX(2, nx + 2 * bx, ny + 2 * by, &phi_old_1, &phi_old_2);

	/* mirror the boundaries */
	mirror_bounds_2d(f_c1, nx, ny, bx, by);
	mirror_bounds_2d(f_c2, nx, ny, bx, by);
	mirror_bounds_2d(f_c3, nx, ny, bx, by);

	/* initialise phi_old */
	copy_matrix_2d(phi1, phi_old_1, nx, ny, bx, by);
	copy_matrix_2d(phi2, phi_old_2, nx, ny, bx, by);

	/* compute time savers */
	hx2 = (2.0 * hx);
	hy2 = (2.0 * hy);
	hx_2 = (hx * hx);
	hy_2 = (hy * hy);

	level_set_to_image(phi1, phi2, nx, ny, bx, by, out);
	if (draw)
		handleDraw();

	c_11_chan_1 = c_12_chan_1 = c_21_chan_1 = c_22_chan_1 = 0.0;
	c_11_chan_2 = c_12_chan_2 = c_21_chan_2 = c_22_chan_2 = 0.0;
	c_11_chan_3 = c_12_chan_3 = c_21_chan_3 = c_22_chan_3 = 0.0;


	/* time stepping */
	for (t = 0.0; Q > tau * hx * hy; t += tau) { //t < T; t += tau){ //

//		printf("\nt: %f, T: %f, mu: %f, nu: %f\n", t, T, mu, nu);

		/* reinitialise the level set function */
		signed_distance(phi1, nx, ny, bx, by);
		signed_distance(phi2, nx, ny, bx, by);

		/* mirror the boundaries */
		mirror_bounds_2d(phi1, nx, ny, bx, by);
		mirror_bounds_2d(phi2, nx, ny, bx, by);
		mirror_bounds_2d(phi_old_1, nx, ny, bx, by);
		mirror_bounds_2d(phi_old_2, nx, ny, bx, by);

		/* initialise */
		old_c_11_chan_1 = c_11_chan_1;
		old_c_12_chan_1 = c_12_chan_1;
		old_c_21_chan_1 = c_21_chan_1;
		old_c_22_chan_1 = c_22_chan_1;

		old_c_11_chan_2 = c_11_chan_2;
		old_c_12_chan_2 = c_12_chan_2;
		old_c_21_chan_2 = c_21_chan_2;
		old_c_22_chan_2 = c_22_chan_2;

		old_c_11_chan_3 = c_11_chan_3;
		old_c_12_chan_3 = c_12_chan_3;
		old_c_21_chan_3 = c_21_chan_3;
		old_c_22_chan_3 = c_22_chan_3;

		old_c_11_chan_4 = c_11_chan_4;
		old_c_12_chan_4 = c_12_chan_4;
		old_c_21_chan_4 = c_21_chan_4;
		old_c_22_chan_4 = c_22_chan_4;

		c_11_chan_1 = c_12_chan_1 = c_21_chan_1 = c_22_chan_1 = 0.0;
		c_11_chan_2 = c_12_chan_2 = c_21_chan_2 = c_22_chan_2 = 0.0;
		c_11_chan_3 = c_12_chan_3 = c_21_chan_3 = c_22_chan_3 = 0.0;
		c_11_chan_4 = c_12_chan_4 = c_21_chan_4 = c_22_chan_4 = 0.0;
		area_11 = area_12 = area_21 = area_22 = 0.0;

		/* update the mean values of both regions */
		for (i = bx; i < nx + bx; i++)
			for (j = by; j < ny + by; j++) {
				/* compute the Heaviside function */
				H1 = heaviside_function(phi_old_1[i][j], epsilon);
				H2 = heaviside_function(phi_old_2[i][j], epsilon);

				c_11_chan_1 += f_c1[i][j] * H1 * H2;
				c_11_chan_2 += f_c2[i][j] * H1 * H2;
				c_11_chan_3 += f_c3[i][j] * H1 * H2;
				c_11_chan_4 += f_c4[i][j] * H1 * H2;
				area_11 += H1 * H2;

				c_12_chan_1 += f_c1[i][j] * H1 * (1.0 - H2);
				c_12_chan_2 += f_c2[i][j] * H1 * (1.0 - H2);
				c_12_chan_3 += f_c3[i][j] * H1 * (1.0 - H2);
				c_12_chan_4 += f_c4[i][j] * H1 * (1.0 - H2);
				area_12 += (H1) * (1.0 - H2);

				c_21_chan_1 += f_c1[i][j] * (1.0 - H1) * H2;
				c_21_chan_2 += f_c2[i][j] * (1.0 - H1) * H2;
				c_21_chan_3 += f_c3[i][j] * (1.0 - H1) * H2;
				c_21_chan_4 += f_c4[i][j] * (1.0 - H1) * H2;
				area_21 += (1.0 - H1) * (H2);

				c_22_chan_1 += f_c1[i][j] * (1.0 - H1) * (1.0 - H2);
				c_22_chan_2 += f_c2[i][j] * (1.0 - H1) * (1.0 - H2);
				c_22_chan_3 += f_c3[i][j] * (1.0 - H1) * (1.0 - H2);
				c_22_chan_4 += f_c4[i][j] * (1.0 - H1) * (1.0 - H2);
				area_22 += (1.0 - H1) * (1.0 - H2);
			}
		c_11_chan_1 /= area_11;
		c_11_chan_2 /= area_11;
		c_11_chan_3 /= area_11;
		c_11_chan_4 /= area_11;

		c_12_chan_1 /= area_12;
		c_12_chan_2 /= area_12;
		c_12_chan_3 /= area_12;
		c_12_chan_4 /= area_12;

		c_21_chan_1 /= area_21;
		c_21_chan_2 /= area_21;
		c_21_chan_3 /= area_21;
		c_21_chan_4 /= area_21;

		c_22_chan_1 /= area_22;
		c_22_chan_2 /= area_22;
		c_22_chan_3 /= area_22;
		c_22_chan_4 /= area_22;




//		printf("\n chan1: c_11: %.2f, c_12: %.2f, c_21: %.2f, c_22: %.2f", c_11_chan_1, c_12_chan_1, c_21_chan_1, c_22_chan_1);
//		printf("\n chan2: c_11: %.2f, c_12: %.2f, c_21: %.2f, c_22: %.2f", c_11_chan_2, c_12_chan_2, c_21_chan_2, c_22_chan_2);
//		printf("\n chan3: c_11: %.2f, c_12: %.2f, c_21: %.2f, c_22: %.2f\n", c_11_chan_3, c_12_chan_3, c_21_chan_3, c_22_chan_3);

//=============================================== PHI 1 ====================================================================
		/* compute the coefficients that depend on the values of the previous time step */
		double newVar = tau / hx * hx;

		double newVar2;
		double C_dem;

		for (i = bx; i < nx + bx; i++)
			for (j = by; j < ny + by; j++) {

				a = (phi_old_1[i + 1][j] - phi_old_1[i][j]) / hx;
				b = (phi_old_1[i][j + 1] - phi_old_1[i][j - 1]) / hy2;
				if ((a == 0.0) && (b == 0.0))
					phi1_C_1[i][j] = 0.0;
				else
					phi1_C_1[i][j] = 1.0 / sqrt(a * a + b * b);

				a = (phi_old_1[i][j] - phi_old_1[i - 1][j]) / hx;
				b = (phi_old_1[i - 1][j + 1] - phi_old_1[i - 1][j - 1]) / hy2;
				if ((a == 0.0) && (b == 0.0))
					phi1_C_2[i][j] = 0.0;
				else
					phi1_C_2[i][j] = 1.0 / sqrt(a * a + b * b);

				a = (phi_old_1[i][j + 1] - phi_old_1[i][j]) / hy;
				b = (phi_old_1[i + 1][j] - phi_old_1[i - 1][j]) / hx2;
				if ((a == 0.0) && (b == 0.0))
					phi1_C_3[i][j] = 0.0;
				else
					phi1_C_3[i][j] = 1.0 / sqrt(a * a + b * b);

				a = (phi_old_1[i][j] - phi_old_1[i][j - 1]) / hy;
				b = (phi_old_1[i + 1][j - 1] - phi_old_1[i - 1][j - 1]) / hx2;
				if ((a == 0.0) && (b == 0.0))
					phi1_C_4[i][j] = 0.0;
				else
					phi1_C_4[i][j] = 1.0 / sqrt(a * a + b * b);


				delta1[i][j] = delta_function(phi_old_1[i][j], epsilon);

				sum1 = ((f_c1[i][j] - c_11_chan_1)* (f_c1[i][j] - c_11_chan_1)) + (f_c2[i][j] - c_11_chan_2)* (f_c2[i][j] - c_11_chan_2) +(f_c3[i][j] - c_11_chan_3)* (f_c3[i][j] - c_11_chan_3) +(f_c4[i][j] - c_11_chan_4)* (f_c4[i][j] - c_11_chan_4);
				sum2 = ((f_c1[i][j] - c_12_chan_1)* (f_c1[i][j] - c_12_chan_1)) + (f_c2[i][j] - c_12_chan_2)* (f_c2[i][j] - c_12_chan_2) +(f_c3[i][j] - c_12_chan_3)* (f_c3[i][j] - c_12_chan_3) +(f_c4[i][j] - c_12_chan_4)* (f_c4[i][j] - c_12_chan_4);
				sum3 = ((f_c1[i][j] - c_21_chan_1)* (f_c1[i][j] - c_21_chan_1)) + (f_c2[i][j] - c_21_chan_2)* (f_c2[i][j] - c_21_chan_2) +(f_c3[i][j] - c_21_chan_3)* (f_c3[i][j] - c_21_chan_3) +(f_c4[i][j] - c_21_chan_4)* (f_c4[i][j] - c_21_chan_4);
				sum4 = ((f_c1[i][j] - c_22_chan_1)* (f_c1[i][j] - c_22_chan_1)) + (f_c2[i][j] - c_22_chan_2)* (f_c2[i][j] - c_22_chan_2) +(f_c3[i][j] - c_22_chan_3)* (f_c3[i][j] - c_22_chan_3) +(f_c4[i][j] - c_22_chan_4)* (f_c4[i][j] - c_22_chan_4);

				sum1 = lambda11 * sum1 * (H2);
				sum2 = lambda12 * sum2 * (1.0 - H2);
				sum3 = lambda21 * sum3 * (H2);
				sum4 = lambda22 * sum4 * (1.0 - H2);

				rhs1[i][j] = tau * delta1[i][j] * (- sum1 - sum2 + sum3 + sum4);

				C_dem = (1.0+ newVar * delta1[i][j] * mu* (phi1_C_1[i][j] + phi1_C_2[i][j] + phi1_C_3[i][j] + phi1_C_4[i][j]));

				newVar2 = phi1_C_1[i][j] * phi_old_1[i + 1][j]
						+ phi1_C_2[i][j] * phi_old_1[i - 1][j]
						+ phi1_C_3[i][j] * phi_old_1[i][j + 1]
						+ phi1_C_4[i][j] * phi_old_1[i][j - 1];

				phi1[i][j] = (1.0 - omega) * phi1[i][j]+ ((omega)* (phi_old_1[i][j]+ (newVar * delta1[i][j] * mu * newVar2)+ rhs1[i][j]));
				phi1[i][j] = phi1[i][j] / C_dem;

			}

//=============================================== PHI 2 ====================================================================
		for (i = bx; i < nx + bx; i++)
			for (j = by; j < ny + by; j++) {

				a = (phi_old_2[i + 1][j] - phi_old_2[i][j]) / hx;
				b = (phi_old_2[i][j + 1] - phi_old_2[i][j - 1]) / hy2;
				if ((a == 0.0) && (b == 0.0))
					phi2_C_1[i][j] = 0.0;
				else
					phi2_C_1[i][j] = 1.0 / sqrt(a * a + b * b);

				a = (phi_old_2[i][j] - phi_old_2[i - 1][j]) / hx;
				b = (phi_old_2[i - 1][j + 1] - phi_old_2[i - 1][j - 1]) / hy2;
				if ((a == 0.0) && (b == 0.0))
					phi2_C_2[i][j] = 0.0;
				else
					phi2_C_2[i][j] = 1.0 / sqrt(a * a + b * b);

				a = (phi_old_2[i][j + 1] - phi_old_2[i][j]) / hy;
				b = (phi_old_2[i + 1][j] - phi_old_2[i - 1][j]) / hx2;
				if ((a == 0.0) && (b == 0.0))
					phi2_C_3[i][j] = 0.0;
				else
					phi2_C_3[i][j] = 1.0 / sqrt(a * a + b * b);

				a = (phi_old_2[i][j] - phi_old_2[i][j - 1]) / hy;
				b = (phi_old_2[i + 1][j - 1] - phi_old_2[i - 1][j - 1]) / hx2;
				if ((a == 0.0) && (b == 0.0))
					phi2_C_4[i][j] = 0.0;
				else
					phi2_C_4[i][j] = 1.0 / sqrt(a * a + b * b);


				delta2[i][j] = delta_function(phi_old_2[i][j], epsilon);

				sum1 = ((f_c1[i][j] - c_11_chan_1)* (f_c1[i][j] - c_11_chan_1)) + (f_c2[i][j] - c_11_chan_2)* (f_c2[i][j] - c_11_chan_2) +(f_c3[i][j] - c_11_chan_3)* (f_c3[i][j] - c_11_chan_3) +(f_c4[i][j] - c_11_chan_4)* (f_c4[i][j] - c_11_chan_4);
				sum2 = ((f_c1[i][j] - c_12_chan_1)* (f_c1[i][j] - c_12_chan_1)) + (f_c2[i][j] - c_12_chan_2)* (f_c2[i][j] - c_12_chan_2) +(f_c3[i][j] - c_12_chan_3)* (f_c3[i][j] - c_12_chan_3) +(f_c4[i][j] - c_12_chan_4)* (f_c4[i][j] - c_12_chan_4);
				sum3 = ((f_c1[i][j] - c_21_chan_1)* (f_c1[i][j] - c_21_chan_1)) + (f_c2[i][j] - c_21_chan_2)* (f_c2[i][j] - c_21_chan_2) +(f_c3[i][j] - c_21_chan_3)* (f_c3[i][j] - c_21_chan_3) +(f_c4[i][j] - c_21_chan_4)* (f_c4[i][j] - c_21_chan_4);
				sum4 = ((f_c1[i][j] - c_22_chan_1)* (f_c1[i][j] - c_22_chan_1)) + (f_c2[i][j] - c_22_chan_2)* (f_c2[i][j] - c_22_chan_2) +(f_c3[i][j] - c_22_chan_3)* (f_c3[i][j] - c_22_chan_3) +(f_c4[i][j] - c_22_chan_4)* (f_c4[i][j] - c_22_chan_4);

				sum1 = lambda11 * sum1 * (H1);
				sum2 = lambda12 * sum2 * (H1);
				sum3 = lambda21 * sum3 * (1.0 - H1);
				sum4 = lambda22 * sum4 * (1.0 - H1);


				rhs2[i][j] = tau * delta2[i][j] * (- sum1 + sum2 - sum3 + sum4);

				C_dem = (1.0 + newVar * delta2[i][j] * mu *
						(phi2_C_1[i][j] + phi2_C_2[i][j]
					   + phi2_C_3[i][j] + phi2_C_4[i][j]));

				newVar2 = phi2_C_1[i][j] * phi_old_2[i + 1][j]
						+ phi2_C_2[i][j] * phi_old_2[i - 1][j]
						+ phi2_C_3[i][j] * phi_old_2[i][j + 1]
						+ phi2_C_4[i][j] * phi_old_2[i][j - 1];

				phi2[i][j] = (1.0 - omega) * phi_old_2[i][j] + omega * (phi_old_2[i][j] + (newVar * delta2[i][j] * mu * newVar2) + rhs2[i][j]);
				phi2[i][j] = phi2[i][j] / C_dem;

			}
//==========================================================================================================================
		/* stopping criterium */

		double difference_c_chan_1 = fabs((old_c_11_chan_1 + old_c_12_chan_1 + old_c_21_chan_1+ old_c_22_chan_1)
									- (c_11_chan_1 + c_12_chan_1 + c_21_chan_1+ c_22_chan_1));
		double difference_c_chan_2 =fabs((old_c_11_chan_2 + old_c_12_chan_2 + old_c_21_chan_2+ old_c_22_chan_2)
								- (c_11_chan_2 + c_12_chan_2 + c_21_chan_2+ c_22_chan_2));
		double difference_c_chan_3 =fabs((old_c_11_chan_3 + old_c_12_chan_3 + old_c_21_chan_3+ old_c_22_chan_3)
								- (c_11_chan_3 + c_12_chan_3 + c_21_chan_3+ c_22_chan_3));
		double difference_c_chan_4 =fabs((old_c_11_chan_4 + old_c_12_chan_4 + old_c_21_chan_4+ old_c_22_chan_4)
										- (c_11_chan_4 + c_12_chan_4 + c_21_chan_4+ c_22_chan_4));
		double difference_c_sum = (difference_c_chan_1 + difference_c_chan_2+ difference_c_chan_3+ difference_c_chan_4) / 4;



		// END for loop:     for(t = 0.0; Q > tau*hx*hy; t += tau)

//		final_c11 = 0;
//		final_c12 = 10;
//		final_c21 = 110;
//		final_c22 = 255;

		if(boolean_of_colors_hard_coded)
		{
			final_c11 = small_details_gray_level;
			final_c12 = background_gray_level;
			final_c21 = skull_gray_level ;
			final_c22 = tumor_gray_level;
		}
		else
		{
			final_c11 = (c_11_chan_1 + c_11_chan_2 + c_11_chan_3 + c_11_chan_4 ) / 4;
			final_c12 = (c_12_chan_1 + c_12_chan_2 + c_12_chan_3 + c_12_chan_4 ) / 4;
			final_c21 = (c_21_chan_1 + c_21_chan_2 + c_21_chan_3 + c_21_chan_4 ) / 4;
			final_c22 = (c_22_chan_1 + c_22_chan_2 + c_22_chan_3 + c_22_chan_4 ) / 4;
		}


		if (counterX % 100 == 0) {
			printf("%5d) Final = %.2f, %.2f, %.2f, %.2f, diff: %.5f \n",
					counterX, final_c11, final_c12, final_c21, final_c22,difference_c_sum);
		}

		if(regular_evaluation && counterX%regular_evaluation_int==0)
		{
			printf("(%d)--",counterX);
			level_set_to_image(phi1, phi2, nx, ny, bx, by, out);
			post_segmented_image(out, nx, ny, bx, by);
			old_global_f_measure = global_f_measure;
			evaluate(f_post_seg, f_ot, nx, ny, bx, by);
			printf("\n");
		}

		counterX++;
		if (counterX > maximumNoIterations	|| difference_c_sum < minimumDifference) {
			Q = 0;
			printf("Iterations No: %d, difference: %f\n", counterX,difference_c_sum);
		}

		copy_matrix_2d(phi1, phi_old_1, nx, ny, bx, by);
		copy_matrix_2d(phi2, phi_old_2, nx, ny, bx, by);

		/* update visualisation */
		level_set_to_image(phi1, phi2, nx, ny, bx, by, out);
//		if(counterX%100==0)
//		{
//			post_segmented_image(out, nx, ny, bx, by);
//			evaluate(f_post_seg, f_ot, nx, ny, bx, by);
//		}
		if (draw)
			handleDraw();
	}
//===================================================================================================================================
	/* free memory */
	FREE_MATRIX(4, nx + 2 * bx, ny + 2 * by, phi1_C_1, phi1_C_2, phi1_C_3,phi1_C_4);
	FREE_MATRIX(4, nx + 2 * bx, ny + 2 * by, phi2_C_1, phi2_C_2, phi2_C_3,phi2_C_4);
	FREE_MATRIX(2, nx + 2 * bx, ny + 2 * by, delta1, delta2);
	FREE_MATRIX(2, nx + 2 * bx, ny + 2 * by, rhs1, rhs2);
	FREE_MATRIX(2, nx + 2 * bx, ny + 2 * by, phi_old_1, phi_old_2);


};


void post_segmented_image(double **segmented_img, int nx, int ny, int bx,
		int by) {
	double larger_value = 0;
	if (final_c11 > larger_value)
		larger_value = final_c11;
	if (final_c12 > larger_value)
		larger_value = final_c12;
	if (final_c21 > larger_value)
		larger_value = final_c21;
	if (final_c22 > larger_value)
		larger_value = final_c22;

	if(boolean_of_colors_hard_coded==false)
	{
		for (i = bx; i < nx + bx; i++) {
			for (j = by; j < ny + by; j++) {
				if (segmented_img[i][j] == larger_value)
					f_post_seg[i][j] = 255;
				else
					f_post_seg[i][j] = 0;
			}
		}
	}
	else
	{
		for (i = bx; i < nx + bx; i++) {
			for (j = by; j < ny + by; j++) {
				if (segmented_img[i][j] == tumor_gray_level)
					f_post_seg[i][j] = 255;
				else
					f_post_seg[i][j] = 0;
			}
		}
	}

}
void evaluate(double **segmented_img, double **ground_truth, int nx, int ny, int bx, int by) {
	double tp = 0;
	double fp = 0;
	double tn = 0;
	double fn = 0;

	for (i = bx; i < nx + bx; i++) {
		for (j = by; j < ny + by; j++) {
			if (segmented_img[i][j] == 0 && ground_truth[i][j] == 0)
				tn++;
			else if (segmented_img[i][j] == 0 && ground_truth[i][j] == 255)
				fn++;
			else if (segmented_img[i][j] == 255 && ground_truth[i][j] == 0)
				fp++;
			else if (segmented_img[i][j] == 255 && ground_truth[i][j] == 255)
				tp++;
		}
	}

//	double precision = tp / (tp + fp);
//	double recall = tp / (tp + fn);
//	double f_measure = 2 * precision * recall / (precision + recall);


	double precision, recall, f_measure;
	precision = ((tp+fp)==0) ? 0 : tp/(tp+fp);
	recall = ((tp+fn)==0) ? 0 : tp/(tp+fn);
	f_measure = ((precision+recall)==0)? 0 : 2*precision*recall/(precision + recall);
	average_score = average_score + f_measure;

	global_f_measure = f_measure;

//	double dice_score = 2*tp/(tp+fp+tp+fn);

//	printf("tn = %f \n", tn);
//	printf("fn = %f \n", fn);
//	printf("fp = %f \n", fp);
//	printf("tp = %f \n", tp);
	printf("precision = %3.3f, recall = %3.3f, f_measure = %3.3f \n", precision, recall, f_measure);

//	background : black : 0,	foreground : white : 1
//  True positive (TP) : pixels correctly segmented as foreground
//  False positive (FP) : pixels falsely segmented as foreground

//  True negative (TN) : pixels correctly detected as background
//  False negative (FN) : pixels falsely detected as background

//	Precision = TP / TP + FP = AB/A
//	Recall = TP / TP + FN = AB/B
//  A: segmented image, B:ground truth, AB:tp, A:tp+fp, B:tp+fn
}


void LVD(double** input_image, int nx, int ny, int bx, int by, double** output_image)
{
	double array[8];
	double up, down, left, right;
	double up_left, up_right, down_left, down_right;
	/*
	1-----5-----3
	7-----9-----8
	7-----9-----8
	7-----9-----8
	2-----6-----4
	*/

	for (i = bx; i < nx + bx; i++)
	{
		for (j = by; j < ny + by; j++)
		{
			//[1] up left corner
			if(i == bx && j == by)
			{
				right = fabs(input_image[i][j] - input_image[i][j+1]);
				down = fabs(input_image[i][j] - input_image[i+1][j]);
				down_right = fabs(input_image[i][j] - input_image[i+1][j+1]);

				array[0] = right;
				array[1] = down;
				array[2] = down_right;
				array[3] = -1;
				array[4] = -1;
				array[5] = -1;
				array[6] = -1;
				array[7] = -1;

				sort_array(array);
				output_image[i][j] = max_value - min_value;
			}
			//[2] down left corner
			else if(i == nx && j == by)
			{
				right = fabs(input_image[i][j] - input_image[i][j+1]);
				up = fabs(input_image[i][j] - input_image[i-1][j]);
				up_right = fabs(input_image[i][j] - input_image[i-1][j+1]);

				array[0] = right;
				array[1] = up;
				array[2] = up_right;
				array[3] = -1;
				array[4] = -1;
				array[5] = -1;
				array[6] = -1;
				array[7] = -1;

				sort_array(array);
				output_image[i][j] = max_value - min_value;
			}
			//3 up right corner
			else if(i == bx && j == ny)
			{
				left = fabs(input_image[i][j] - input_image[i][j-1]);
				down = fabs(input_image[i][j] - input_image[i+1][j]);
				down_left = fabs(input_image[i][j] - input_image[i+1][j-1]);

				array[0] = left;
				array[1] = down;
				array[2] = down_left;
				array[3] = -1;
				array[4] = -1;
				array[5] = -1;
				array[6] = -1;
				array[7] = -1;

				sort_array(array);
				output_image[i][j] = max_value - min_value;
			}
			//4 down right corner
			else if(i == nx && j == ny)
			{
				left = fabs(input_image[i][j] - input_image[i][j-1]);
				up = fabs(input_image[i][j] - input_image[i-1][j]);
				up_left = fabs(input_image[i][j] - input_image[i-1][j-1]);

				array[0] = left;
				array[1] = up;
				array[2] = up_left;
				array[3] = -1;
				array[4] = -1;
				array[5] = -1;
				array[6] = -1;
				array[7] = -1;

				sort_array(array);
				output_image[i][j] = max_value - min_value;
			}

			//5 up row
			else if(i == bx)
			{
				right = fabs(input_image[i][j] - input_image[i][j+1]);
				down = fabs(input_image[i][j] - input_image[i+1][j]);
				down_right = fabs(input_image[i][j] - input_image[i+1][j+1]);
				left = fabs(input_image[i][j] - input_image[i][j-1]);
				down_left = fabs(input_image[i][j] - input_image[i+1][j-1]);

				array[0] = right;
				array[1] = down;
				array[2] = down_right;
				array[3] = left;
				array[4] = down_left;
				array[5] = -1;
				array[6] = -1;
				array[7] = -1;

				sort_array(array);
				output_image[i][j] = max_value - min_value;
			}
			//6 down row
			else if(i == nx)
			{
				right = fabs(input_image[i][j] - input_image[i][j+1]);
				up = fabs(input_image[i][j] - input_image[i-1][j]);
				up_right = fabs(input_image[i][j] - input_image[i-1][j+1]);
				left = fabs(input_image[i][j] - input_image[i][j-1]);
				up_left = fabs(input_image[i][j] - input_image[i-1][j-1]);

				array[0] = right;
				array[1] = up;
				array[2] = up_right;
				array[3] = left;
				array[4] = up_left;
				array[5] = -1;
				array[6] = -1;
				array[7] = -1;

				sort_array(array);
				output_image[i][j] = max_value - min_value;
			}
			//7 left column
			else if(j == by)
			{
				right = fabs(input_image[i][j] - input_image[i][j+1]);
				down = fabs(input_image[i][j] - input_image[i+1][j]);
				down_right = fabs(input_image[i][j] - input_image[i+1][j+1]);
				up = fabs(input_image[i][j] - input_image[i-1][j]);
				up_right = fabs(input_image[i][j] - input_image[i-1][j+1]);

				array[0] = right;
				array[1] = down;
				array[2] = down_right;
				array[3] = up;
				array[4] = up_right;
				array[5] = -1;
				array[6] = -1;
				array[7] = -1;

				sort_array(array);
				output_image[i][j] = max_value - min_value;
			}
			//8 right column
			else if(j == ny)
			{
				left = fabs(input_image[i][j] - input_image[i][j-1]);
				down = fabs(input_image[i][j] - input_image[i+1][j]);
				down_left = fabs(input_image[i][j] - input_image[i+1][j-1]);
				up = fabs(input_image[i][j] - input_image[i-1][j]);
				up_left = fabs(input_image[i][j] - input_image[i-1][j-1]);

				array[0] = left;
				array[1] = down;
				array[2] = down_left;
				array[3] = up;
				array[4] = up_left;
				array[5] = -1;
				array[6] = -1;
				array[7] = -1;

				sort_array(array);
				output_image[i][j] = max_value - min_value;
			}


			//9 centered pixel
			else
			{
				down = fabs(input_image[i][j] - input_image[i+1][j]);
				up = fabs(input_image[i][j] - input_image[i-1][j]);
				right = fabs(input_image[i][j] - input_image[i][j+1]);
				left = fabs(input_image[i][j] - input_image[i][j-1]);
				down_right = fabs(input_image[i][j] - input_image[i+1][j+1]);
				down_left = fabs(input_image[i][j] - input_image[i+1][j-1]);
				up_right = fabs(input_image[i][j] - input_image[i-1][j+1]);
				up_left = fabs(input_image[i][j] - input_image[i-1][j-1]);

				array[0] = down;
				array[1] = up;
				array[2] = right;
				array[3] = left;
				array[4] = down_right;
				array[5] = down_left;
				array[6] = up_right;
				array[7] = up_left;

				sort_array(array);
//				double sum = 0;
//				for (int var = 0;  var < 8;  var++)
//					sum=sum+array[var];
//				if(sum>0){
//					for (int var = 0;  var < 8;  var++)
//						printf("%2.2f  ", array[var]);
//					printf("\n");
//
//				printf("max_value = %2.2f\n",max_value);
//				printf("min_value = %2.2f\n", min_value);
//				}
				output_image[i][j] = max_value - min_value;
			}
		}
	}
}

void sort_array(double array[8])
{
	max_value = -1;
	min_value = 1000;
	for (int var = 0; var < 8; var++)
	{
		if(array[var]==-1)
			continue;
		if(array[var] >= max_value)
			max_value = array[var];
		if(array[var] <= min_value)
			min_value = array[var];
	}
}


void extract_textural_features(double** input_image, int nx, int ny, int bx, int by, int counter_of_text_feature, int theta, int window_size)
{
//	printf("begin of extract_textural_features \n");


	for (int mos = 120; mos < 125; mos++)
	{
		for (int most = 90; most < 95; most++)
		{
			printf("[%d][%d]%.0lf  ", mos, most, input_image[mos][most]);
		}
		printf("\n");
	}
	printf("\n\n\n");

	int value_to_round = 7;
//	int window_size = 7;
	int half_window_size = window_size / 2;
	int Ng = value_to_round+1;

	printf("window_size = %d , half_window_size = %d \n",window_size,half_window_size);

	char *string_textural_feature_all = (char*)malloc(100);
	char *string_textural_feature_theta = (char*)malloc(100);
	char *string_textural_feature_name = (char*)malloc(100);

	// select which textural feature to calculate
//	counter_temp = 1;
//	theta = 0;

	double a, b;
	double sum;
	double mean_x, mean_y;
	double std_x, std_y;
	double mean, variance;
	double angular_second_moment;
	double contrast;
	double correlation;
	double sum_of_squares;
	double inverse_different_moment;
	double sum_of_average;
	double sum_of_variance;
	double sum_of_entropy;
	double entropy;
	double difference_of_variance;
	double difference_of_entropy;
	double information_measures_of_correlation_12;
	double information_measures_of_correlation_13;
	double maximal_correlation_coefficient;


	switch (counter_of_text_feature)
	{
		case 1:	strcpy(string_textural_feature_name, "(1)angular_second_moment"); break;
		case 2:	strcpy(string_textural_feature_name, "(2)contrast"); break;
		case 3:	strcpy(string_textural_feature_name, "(3)correlation"); break;
		case 4:	strcpy(string_textural_feature_name, "(4)sum_of_squares"); 	break;
		case 5:	strcpy(string_textural_feature_name, "(5)inverse_different_moment");  break;
		case 6:	strcpy(string_textural_feature_name, "(6)sum_of_average"); 	break;
		case 7:	strcpy(string_textural_feature_name, "(7)sum_of_variance"); break;
		case 8:	strcpy(string_textural_feature_name, "(8)sum_of_entropy"); 	break;
		case 9:	strcpy(string_textural_feature_name, "(9)entropy"); break;
		case 10: strcpy(string_textural_feature_name,"(10)difference_of_variance"); break;
		case 11: strcpy(string_textural_feature_name, "(11)difference_of_entropy"); break;
		case 12: strcpy(string_textural_feature_name, "(12)information_measures_of_correlation"); break;
		case 13: strcpy(string_textural_feature_name, "(13)information_measures_of_correlation"); break;
		case 14: strcpy(string_textural_feature_name, "(14)maximal_correlation_coefficient"); break;
		default: break;
	}
	sprintf(string_textural_feature_theta, "__theta_%d_", theta);
//	sprintf(string_textural_feature_theta, "__w_%d_", window_size);
	strcpy(string_textural_feature_all, string_textural_feature_name);
	strcat(string_textural_feature_all, string_textural_feature_theta);

	double** newImageResult;
	ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &newImageResult);

	// create new image in range of NEW value_to_round (Here range 0-7)
	double** temp_rounded_value;
	ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &temp_rounded_value);
	for (int i = bx; i < nx+bx; i++)
		for (int j = by; j < ny+by; j++)
			temp_rounded_value[i][j] = round(value_to_round * input_image[i][j] / 255);


	for (int mos = 120; mos < 125; mos++)
	{
		for (int most = 90; most < 96; most++)
		{
			printf("%.0lf  ", temp_rounded_value[mos][most]);
		}
		printf("\n");
	}
	printf("\n\n\n");

	// extract small patches
	for (int i = bx + half_window_size ; i < nx - half_window_size ; i++)
	{
		for (int j = by + half_window_size ; j < ny - half_window_size ; j++)
		{
		// main pixel is temp[i][j]
//		printf("[%d][%d]\n",i,j);
		double** temp_GLCmatrix;
		ALLOC_MATRIX(1, value_to_round+1, value_to_round+1, &temp_GLCmatrix);
		sum = 0;

//		printf("ALLOC_MATRIX \n");
		for (int k = i-half_window_size ; k <= i+half_window_size ; k++)
		{
			for (int q = j-half_window_size ; q <= j+half_window_size ; q++)
			{
				if(theta==0)
				{
					a = temp_rounded_value[k][q];
					b = temp_rounded_value[k][q+1];
				}
				else if(theta==45)
				{
					a = temp_rounded_value[k][q];
					b = temp_rounded_value[k-1][q+1];
				}
				else if(theta==90)
				{
					a = temp_rounded_value[k][q];
					b = temp_rounded_value[k-1][q];
				}
				else if(theta==135)
				{
					a = temp_rounded_value[k][q];
					b = temp_rounded_value[k-1][q-1];
				}
				if(a!=0 || b!=0)
				{
					int inta = (int)a;
					int intb = (int)b;
					temp_GLCmatrix[inta][intb]++;
					sum++;
				}
			}
		}

//		if(i==92 && j==123)
//		{
//			for (int mos = 0; mos < value_to_round+1; mos++)
//			{
//				for (int most = 0; most < value_to_round+1; most++)
//				{
//					printf("%.0lf  ", temp_GLCmatrix[mos][most]);
//				}
//				printf("\n");
//			}
//			printf("\n\n\n");
//		}


//		printf(" 1 \n");
		// calculate feature of small temp_GLCmatrix
		mean_x = 0;
		mean_y = 0;
		std_x = 0;
		std_y = 0;
		mean = 0;
		variance = 0;

		angular_second_moment = 0;
		contrast = 0;
		correlation = 0;
		sum_of_squares = 0;
		inverse_different_moment = 0;
		sum_of_average = 0;
		sum_of_variance = 0;
		sum_of_entropy = 0;
		entropy = 0;
		difference_of_variance = 0;
		difference_of_entropy = 0;
		information_measures_of_correlation_12 = 0;
		information_measures_of_correlation_13 = 0;
		maximal_correlation_coefficient = 0;

		// normalize the matrix AND calculate mean and standard variation
		for(int m = 0 ; m <= value_to_round ; m++)
		{
			for(int n = 0 ; n <= value_to_round ; n++)
			{
				if(sum!=0)
					temp_GLCmatrix[m][n] = temp_GLCmatrix[m][n] / sum;
				mean = mean + (m*n*temp_GLCmatrix[m][n]);
				mean_x = mean_x + (m*temp_GLCmatrix[m][n]);
				mean_y = mean_y + (n*temp_GLCmatrix[m][n]);
			}
		}
		for(int m = 0 ; m <= value_to_round ; m++)
		{
			for(int n = 0 ; n <= value_to_round ; n++)
			{
				std_x = std_x + ((m-mean_x)*(m-mean_x)*temp_GLCmatrix[m][n]);
				std_y = std_y + ((n-mean_y)*(n-mean_y)*temp_GLCmatrix[m][n]);
			}
		}



		switch (counter_of_text_feature)
		{
			case 1:
			{
//				printf(" case 1 \n");
//				(1) Angular Second Moment
				for(int m = 0 ; m <= value_to_round ; m++)
				{
					for(int n = 0 ; n <= value_to_round ; n++)
					{
						if(temp_GLCmatrix[m][n]!=0)
							angular_second_moment = angular_second_moment + (temp_GLCmatrix[m][n] * temp_GLCmatrix[m][n]);
					}
				}
				angular_second_moment = floor(angular_second_moment * 256);
				newImageResult[i][j] = angular_second_moment;
				break;
			}
//-----------------------------------------------------------------------------------------------
			case 2:
			{
//				(2) contrast
				for(int v = 0 ; v < value_to_round ; v++)
				{
					for(int m = 0 ; m <= value_to_round ; m++)
					{
						for(int n = 0 ; n <= value_to_round ; n++)
						{
							int m2 = m+1;
							int n2 = n+1;
							if(abs(m2-n2) == v)
							{
								if(temp_GLCmatrix[m][n]!=0)
									contrast = contrast + (v*v*temp_GLCmatrix[m][n]);
							}
						}
					}
				}
				contrast = floor(contrast * 256);
				newImageResult[i][j] = contrast;
				break;
			}
//-----------------------------------------------------------------------------------------------
			case 3:
			{
//			(3) Correlation
				for(int m = 0 ; m <= value_to_round ; m++)
				{
					for(int n = 0 ; n <= value_to_round ; n++)
					{
						correlation = correlation + (m*n*temp_GLCmatrix[m][n]);
					}
				}

				if(std_x*std_y != 0)
					correlation = (correlation - (mean_x*mean_y)) / (std_x*std_y);
//				else
//					correlation = (correlation - (mean_x*mean_y)) / (std_x*std_y);

				correlation = floor(correlation * 256 / 16);
				newImageResult[i][j] = correlation;
				break;
			}
	//-----------------------------------------------------------------------------------------------
			case 4:
			{
	//			(4) Sum of Squares: Variance
				for(int m = 0 ; m <= value_to_round ; m++)
				{
					for(int n = 0 ; n <= value_to_round ; n++)
					{
						if(temp_GLCmatrix[m][n]!=0)
							sum_of_squares = sum_of_squares + ((n-mean_y)*(n-mean_y)*temp_GLCmatrix[m][n]);
					}
				}
				sum_of_squares = floor(sum_of_squares * 256);
				newImageResult[i][j] = sum_of_squares;
				break;
			}
	//-----------------------------------------------------------------------------------------------
			case 5:
			{
//				(5) Inverse Different Moment
				for(int m = 0 ; m <= value_to_round ; m++)
				{
					for(int n = 0 ; n <= value_to_round ; n++)
					{
						inverse_different_moment = inverse_different_moment + (temp_GLCmatrix[m][n] / (1 + (m-n)*(m-n)));
					}
				}
				inverse_different_moment = floor(inverse_different_moment * 256);
				newImageResult[i][j] = inverse_different_moment;

				break;
			}
//-----------------------------------------------------------------------------------------------
			case 6:
			{
//				(6) Sum of Average
				double p_x_plus_y[2*Ng];
				for(int v = 1 ; v < 2*Ng ; v++)
				{
					p_x_plus_y[v] = 0;
					for(int m = 0 ; m <= value_to_round ; m++)
					{
						for(int n = 0 ; n <= value_to_round ; n++)
						{
							if(m+n==v)
								p_x_plus_y[v] = p_x_plus_y[v] + temp_GLCmatrix[m][n];
						}
					}
					sum_of_average = sum_of_average + ((v+1)*p_x_plus_y[v]);
				}
				sum_of_average = floor(sum_of_average * 256/16);
				newImageResult[i][j] = sum_of_average;
				break;
			}
//-----------------------------------------------------------------------------------------------
			case 7:
			{
//				(7) Sum of Variance
				double p_x_plus_y[2*Ng];
				for(int v = 1 ; v < 2*Ng ; v++)
				{
					p_x_plus_y[v] = 0;
					for(int m = 0 ; m <= value_to_round ; m++)
					{
						for(int n = 0 ; n <= value_to_round ; n++)
						{
							if(m+n==v)
								p_x_plus_y[v] = p_x_plus_y[v] + temp_GLCmatrix[m][n];
						}
					}
					if(p_x_plus_y[v]!=0)
						sum_of_entropy = sum_of_entropy + (p_x_plus_y[v] * log(p_x_plus_y[v]));
				}
				for(int v = 1 ; v < 2*Ng ; v++)
					sum_of_variance = sum_of_variance + ((v+1-sum_of_entropy)*(v+1-sum_of_entropy)*p_x_plus_y[v]);
				newImageResult[i][j] = sum_of_variance;
				break;
			}
//-----------------------------------------------------------------------------------------------
			case 8:
			{
//				(8) Sum of Entropy
				double p_x_plus_y[2*Ng];
				for(int v = 1 ; v < 2*Ng ; v++)
				{
					p_x_plus_y[v] = 0;
					for(int m = 0 ; m <= value_to_round ; m++)
					{
						for(int n = 0 ; n <= value_to_round ; n++)
						{
							if(m+n==v)
								p_x_plus_y[v] = p_x_plus_y[v] + temp_GLCmatrix[m][n];
						}
					}
					if(p_x_plus_y[v]!=0)
						sum_of_entropy = sum_of_entropy + (p_x_plus_y[v] * log(p_x_plus_y[v]));
				}
				sum_of_entropy = floor(-sum_of_entropy * 256);
				newImageResult[i][j] = sum_of_entropy;
				break;
			}
//-----------------------------------------------------------------------------------------------
			case 9:
			{
//				(9) entropy
				for(int m = 0 ; m <= value_to_round ; m++)
				{
					for(int n = 0 ; n <= value_to_round ; n++)
					{
						if(temp_GLCmatrix[m][n]!=0)
							entropy = entropy + (temp_GLCmatrix[m][n] * log(temp_GLCmatrix[m][n]));
					}
				}
				entropy = floor(-entropy * 256);
				newImageResult[i][j] = entropy;
				break;
			}
//-----------------------------------------------------------------------------------------------
			case 10:
			{
//				(10) Difference of Variance
				double p_x_plus_y[2*Ng];
				double meanTemp=0;
				for(int v = 1 ; v < 2*Ng ; v++)
				{
					p_x_plus_y[v] = 0;
					for(int m = 0 ; m <= value_to_round ; m++)
					{
						for(int n = 0 ; n <= value_to_round ; n++)
						{
							if(m+n==v)
								p_x_plus_y[v] = p_x_plus_y[v] + temp_GLCmatrix[m][n];
						}
					}
					meanTemp = meanTemp + p_x_plus_y[v];
				}
				meanTemp = meanTemp / (2*Ng);
				for(int v = 1 ; v < 2*Ng ; v++)
					difference_of_variance = difference_of_variance + ((p_x_plus_y[v]-meanTemp)*(p_x_plus_y[v]-meanTemp));
				difference_of_variance = difference_of_variance / (2*Ng-1);
				difference_of_variance = floor(difference_of_variance * 256 * 256 / 16);
				newImageResult[i][j] = difference_of_variance;
				break;
			}
//-----------------------------------------------------------------------------------------------
			case 11:
			{
//				(11) Difference of Entropy
				double p_x_minus_y[Ng-1];
				for(int v = 0 ; v <= Ng-1 ; v++)
				{
					p_x_minus_y[v] = 0;
					for(int m = 0 ; m <= value_to_round ; m++)
					{
						for(int n = 0 ; n <= value_to_round ; n++)
						{
							if(abs(m-n)==v)
								p_x_minus_y[v] = p_x_minus_y[v] + temp_GLCmatrix[m][n];
						}
					}
					if(p_x_minus_y[v]!=0)
						difference_of_entropy = difference_of_entropy + (p_x_minus_y[v] * log(p_x_minus_y[v]));
				}
				newImageResult[i][j] = -difference_of_entropy * 256;
				break;
			}
//-----------------------------------------------------------------------------------------------
			case 12:
			{
//				(12), (13)		Information measures of Correlation
				double p_x[Ng-1];
				double p_y[Ng-1];
				double HX = 0;
				double HY = 0;
				double HXY = 0;
				double HXY1 = 0;
				double HXY2 = 0;

				for(int m = 0 ; m <= Ng-1 ; m++)
				{
					p_x[m] = 0;
					for(int n = 0 ; n <= Ng-1 ; n++)
						p_x[m] = p_x[m] + temp_GLCmatrix[m][n];
				}

				for(int n = 0 ; n <= Ng-1 ; n++)
				{
					p_y[n] = 0;
					for(int m = 0 ; m <= Ng-1 ; m++)
						p_y[n] = p_y[n] + temp_GLCmatrix[m][n];
				}

				for(int m = 0 ; m <= Ng-1 ; m++)
				{
					if(p_x[m]!=0)
						HX = HX + (p_x[m]*log(p_x[m]));
					if(p_y[m]!=0)
						HY = HY + (p_y[m]*log(p_y[m]));
				}


				for(int m = 0 ; m <= Ng-1 ; m++)
					for(int n = 0 ; n <= Ng-1 ; n++)
						if(temp_GLCmatrix[m][n]!=0)
							HXY = HXY + (temp_GLCmatrix[m][n] * log(temp_GLCmatrix[m][n]));
				HXY = -HXY;

				for(int m = 0 ; m <= Ng-1 ; m++)
					for(int n = 0 ; n <= Ng-1  ; n++)
						if(temp_GLCmatrix[m][n]!=0)
							HXY1 = HXY1 + (temp_GLCmatrix[m][n] * log(p_x[m]*p_y[n]));
				HXY1 = -HXY1;

				for(int m = 0 ; m <= Ng-1 ; m++)
					for(int n = 0 ; n <= Ng-1  ; n++)
						if(temp_GLCmatrix[m][n]!=0)
							HXY2 = HXY2 + (p_x[m] * p_y[n] * log(p_x[m]*p_y[n]));
				HXY2 = -HXY2;

				if(max(HX,HY)!=0)
					information_measures_of_correlation_12 = (HXY - HXY1) / (max(HX,HY));
				newImageResult[i][j] = information_measures_of_correlation_12*256;
				break;
			}
//-----------------------------------------------------------------------------------------------
			case 13:
			{
				double p_x[Ng-1];
				double p_y[Ng-1];
				double HX = 0;
				double HY = 0;
				double HXY = 0;
				double HXY1 = 0;
				double HXY2 = 0;

				for(int m = 0 ; m <= Ng-1 ; m++)
				{
					p_x[m]=0;
					for(int n = 0 ; n <= Ng-1 ; n++)
						p_x[m] = p_x[m] + temp_GLCmatrix[m][n];
				}

				for(int n = 0 ; n <= Ng-1 ; n++)
				{
					p_y[n]=0;
					for(int m = 0 ; m <= Ng-1 ; m++)
						p_y[n] = p_y[n] + temp_GLCmatrix[m][n];
				}

				for(int m = 0 ; m <= Ng-1 ; m++)
				{
					if(p_x[m]!=0)
						HX = HX + (p_x[m]*log(p_x[m]));
					if(p_y[m]!=0)
						HY = HY + (p_y[m]*log(p_y[m]));
				}

				for(int m = 0 ; m <= Ng-1 ; m++)
					for(int n = 0 ; n <= Ng-1  ; n++)
						if(temp_GLCmatrix[m][n]!=0)
							HXY = HXY + (temp_GLCmatrix[m][n] * log(temp_GLCmatrix[m][n]));
				HXY = -HXY;

				for(int m = 0 ; m <= Ng-1 ; m++)
					for(int n = 0 ; n <= Ng-1  ; n++)
						if(temp_GLCmatrix[m][n]!=0)
							HXY1 = HXY1 + (temp_GLCmatrix[m][n] * log(p_x[m]*p_y[n]));
				HXY1 = -HXY1;

				for(int m = 0 ; m <= Ng-1 ; m++)
					for(int n = 0 ; n <= Ng-1  ; n++)
						if(temp_GLCmatrix[m][n]!=0)
							HXY2 = HXY2 + (p_x[m] * p_y[n] * log(p_x[m]*p_y[n]));
				HXY2 = -HXY2;

				double HXY_temp = exp(-2.0 * (HXY2 - HXY));
				information_measures_of_correlation_13 = sqrt(abs(1 - HXY_temp));
				newImageResult[i][j] = floor(information_measures_of_correlation_13*256);
				break;
			}
	//-----------------------------------------------------------------------------------------------
			case 14:
			{
	//				(14)	 Maximal Correlation Coefficient

				double p_x[Ng-1];
				double p_y[Ng-1];

				double** Q;
				ALLOC_MATRIX(1, Ng, Ng, &Q);

				for(int m = 0 ; m <= Ng-1 ; m++)
				{
					p_x[m]=0;
					for(int n = 0 ; n <= Ng-1 ; n++)
						p_x[m] = p_x[m] + temp_GLCmatrix[m][n];
				}

				for(int n = 0 ; n <= Ng-1 ; n++)
				{
					p_y[n]=0;
					for(int m = 0 ; m <= Ng-1 ; m++)
						p_y[n] = p_y[n] + temp_GLCmatrix[m][n];
				}

				for(int m = 0 ; m <= Ng-1 ; m++)
				{
					for(int n = 0 ; n <= Ng-1 ; n++)
					{
						for(int v = 0 ; v <= Ng-1 ; v++)
						{
							Q[m][n] = Q[m][n] + ((temp_GLCmatrix[m][v]*temp_GLCmatrix[n][v]) / (p_x[m]*p_y[v]));
						}
					}
				}
				newImageResult[i][j] = floor(maximal_correlation_coefficient*256);

				break;
			}
			default:
				break;
			}

//		printf(" FREE matrix \n");
//		FREE_MATRIX(1, value_to_round+1, value_to_round+1, &temp_GLCmatrix);
		}

	}

	char out_image_str[100];
	strcpy(out_image_str, image_title_str);
	strcat(out_image_str, "__");
	strcat(out_image_str, string_textural_feature_all);
	strcat(out_image_str, ".pgm");

	write_pgm_blank_header(out_image_str, nx, ny);
	write_pgm_data(out_image_str, newImageResult, nx, ny, bx, by);

	free(string_textural_feature_all);
	free(string_textural_feature_theta);
	free(string_textural_feature_name);
}

double** LBP(double** input_image, int nx, int ny, int bx, int by)
{
	double weights[3][3] = { { 1, 2, 4 }, { 128, 0, 8 }, { 64, 32, 16 } };

	double** newImageResult;
	ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &newImageResult);

	for (i = 3; i < nx + bx; i++)
	{
		for (j = 3; j < ny + by; j++)
		{
			double sum=0;
			double** temp_matrix;
			ALLOC_MATRIX(1, 3, 3, &temp_matrix);
			for (int k = i-1 ; k <= i+1 ; k++)
			{
				for (int q = j-1 ; q <= j+1 ; q++)
				{
					temp_matrix[k-i+1][q-j+1] = (input_image[i][j]-input_image[k][q]-25>=0) ?  1 : 0;
				}
			}
			for (int k = 0 ; k <= 2 ; k++)
			{
				for (int q = 0 ; q <= 2 ; q++)
				{
					temp_matrix[k][q] = temp_matrix[k][q] * weights[k][q];
					sum = sum + temp_matrix[k][q];
				}
			}
			if(sum==255)
				sum=0;
			newImageResult[i][j] = sum;
		}
	}
	write_pgm_blank_header("LBP.pgm", nx, ny);
	write_pgm_data("LBP.pgm", newImageResult, nx, ny, bx, by);
	return newImageResult;
}

void new_images_minus(double** d, double** f, double** t1, double** t2, int nx, int ny, int bx, int by)
{
	double** f_minus_d;
	ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &f_minus_d);
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++)
			f_minus_d[i][j] = f[i][j] - d[i][j];
	char new_str_1[200];
	strcpy(new_str_1, image_published_str);
	strcat(new_str_1, "f_minus_d.pgm");
	write_pgm_blank_header(new_str_1, nx, ny);
	write_pgm_data(new_str_1, f_minus_d, nx, ny, bx, by);


	double** f_minus_t1;
	ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &f_minus_t1);
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++)
			f_minus_t1[i][j] = f[i][j] - t1[i][j];
	char new_str_2[200];
	strcpy(new_str_2, image_published_str);
	strcat(new_str_2, "f_minus_t1.pgm");
	write_pgm_blank_header(new_str_2, nx, ny);
	write_pgm_data(new_str_2, f_minus_t1, nx, ny, bx, by);


	double** f_minus_t2;
	ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &f_minus_t2);
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++)
			f_minus_t2[i][j] = f[i][j] - t2[i][j];
	char new_str_3[200];
	strcpy(new_str_3, image_published_str);
	strcat(new_str_3, "f_minus_t2.pgm");
	write_pgm_blank_header(new_str_3, nx, ny);
	write_pgm_data(new_str_3, f_minus_t2, nx, ny, bx, by);


	double** d_minus_t2;
	ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &d_minus_t2);
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++)
			d_minus_t2[i][j] = d[i][j] - t2[i][j];
	char new_str_4[200];
	strcpy(new_str_4, image_published_str);
	strcat(new_str_4, "d_minus_t2.pgm");
	write_pgm_blank_header(new_str_4, nx, ny);
	write_pgm_data(new_str_4, d_minus_t2, nx, ny, bx, by);


	double** d_minus_t1;
	ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &d_minus_t1);
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++)
			d_minus_t1[i][j] = d[i][j] - t1[i][j];
	char new_str_5[200];
	strcpy(new_str_5, image_published_str);
	strcat(new_str_5, "d_minus_t1.pgm");
	write_pgm_blank_header(new_str_5, nx, ny);
	write_pgm_data(new_str_5, d_minus_t1, nx, ny, bx, by);


	double** t1_minus_t2;
	ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &t1_minus_t2);
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++)
			t1_minus_t2[i][j] = t1[i][j] - t2[i][j];
	char new_str_6[200];
	strcpy(new_str_6, image_published_str);
	strcat(new_str_6, "t1_minus_t2.pgm");
	write_pgm_blank_header(new_str_6, nx, ny);
	write_pgm_data(new_str_6, t1_minus_t2, nx, ny, bx, by);





	double** d_minus_f;
	ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &d_minus_f);
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++)
			d_minus_f[i][j] = d[i][j] - f[i][j];
	char new_str_7[200];
	strcpy(new_str_7, image_published_str);
	strcat(new_str_7, "d_minus_f.pgm");
	write_pgm_blank_header(new_str_7, nx, ny);
	write_pgm_data(new_str_7, d_minus_f, nx, ny, bx, by);


	double** t1_minus_f;
	ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &t1_minus_f);
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++)
			t1_minus_f[i][j] = t1[i][j] - f[i][j];
	char new_str_8[200];
	strcpy(new_str_8, image_published_str);
	strcat(new_str_8, "t1_minus_f.pgm");
	write_pgm_blank_header(new_str_8, nx, ny);
	write_pgm_data(new_str_8, t1_minus_f, nx, ny, bx, by);


	double** t2_minus_f;
	ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &t2_minus_f);
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++)
			t2_minus_f[i][j] = t2[i][j] - f[i][j];
	char new_str_9[200];
	strcpy(new_str_9, image_published_str);
	strcat(new_str_9, "t2_minus_f.pgm");
	write_pgm_blank_header(new_str_9, nx, ny);
	write_pgm_data(new_str_9, t2_minus_f, nx, ny, bx, by);


	double** t2_minus_d;
	ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &t2_minus_d);
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++)
			t2_minus_d[i][j] = t2[i][j] - d[i][j];
	char new_str_10[200];
	strcpy(new_str_10, image_published_str);
	strcat(new_str_10, "t2_minus_d.pgm");
	write_pgm_blank_header(new_str_10, nx, ny);
	write_pgm_data(new_str_10, t2_minus_d, nx, ny, bx, by);


	double** t1_minus_d;
	ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &t1_minus_d);
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++)
			t1_minus_d[i][j] = t1[i][j] - d[i][j];
	char new_str_11[200];
	strcpy(new_str_11, image_published_str);
	strcat(new_str_11, "t1_minus_d.pgm");
	write_pgm_blank_header(new_str_11, nx, ny);
	write_pgm_data(new_str_11, t1_minus_d, nx, ny, bx, by);


	double** t2_minus_t1;
	ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &t2_minus_t1);
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++)
			t2_minus_t1[i][j] = t2[i][j] - t1[i][j];
	char new_str_12[200];
	strcpy(new_str_12, image_published_str);
	strcat(new_str_12, "t2_minus_t1.pgm");
	write_pgm_blank_header(new_str_12, nx, ny);
	write_pgm_data(new_str_12, t2_minus_t1, nx, ny, bx, by);
}

void new_images_plus(double** d, double** f, double** t1, double** t2, int nx, int ny, int bx, int by)
{
	double** f_plus_d;
	ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &f_plus_d);
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++)
			f_plus_d[i][j] = f[i][j] + d[i][j];
	char new_str_1[200];
	strcpy(new_str_1, image_published_str);
	strcat(new_str_1, "f_plus_d.pgm");
	write_pgm_blank_header(new_str_1, nx, ny);
	write_pgm_data(new_str_1, f_plus_d, nx, ny, bx, by);


	double** f_plus_t1;
	ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &f_plus_t1);
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++)
			f_plus_t1[i][j] = f[i][j] + t1[i][j];
	char new_str_2[200];
	strcpy(new_str_2, image_published_str);
	strcat(new_str_2, "f_plus_t1.pgm");
	write_pgm_blank_header(new_str_2, nx, ny);
	write_pgm_data(new_str_2, f_plus_t1, nx, ny, bx, by);


	double** f_plus_t2;
	ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &f_plus_t2);
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++)
			f_plus_t2[i][j] = f[i][j] + t2[i][j];
	char new_str_3[200];
	strcpy(new_str_3, image_published_str);
	strcat(new_str_3, "f_plus_t2.pgm");
	write_pgm_blank_header(new_str_3, nx, ny);
	write_pgm_data(new_str_3, f_plus_t2, nx, ny, bx, by);


	double** d_plus_t2;
	ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &d_plus_t2);
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++)
			d_plus_t2[i][j] = d[i][j] + t2[i][j];
	char new_str_4[200];
	strcpy(new_str_4, image_published_str);
	strcat(new_str_4, "d_plus_t2.pgm");
	write_pgm_blank_header(new_str_4, nx, ny);
	write_pgm_data(new_str_4, d_plus_t2, nx, ny, bx, by);


	double** d_plus_t1;
	ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &d_plus_t1);
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++)
			d_plus_t1[i][j] = d[i][j] + t1[i][j];
	char new_str_5[200];
	strcpy(new_str_5, image_published_str);
	strcat(new_str_5, "d_plus_t1.pgm");
	write_pgm_blank_header(new_str_5, nx, ny);
	write_pgm_data(new_str_5, d_plus_t1, nx, ny, bx, by);


	double** t1_plus_t2;
	ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &t1_plus_t2);
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++)
			t1_plus_t2[i][j] = t1[i][j] + t2[i][j];
	char new_str_6[200];
	strcpy(new_str_6, image_published_str);
	strcat(new_str_6, "t1_plus_t2.pgm");
	write_pgm_blank_header(new_str_6, nx, ny);
	write_pgm_data(new_str_6, t1_plus_t2, nx, ny, bx, by);





	double** d_plus_f;
	ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &d_plus_f);
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++)
			d_plus_f[i][j] = d[i][j] + f[i][j];
	char new_str_7[200];
	strcpy(new_str_7, image_published_str);
	strcat(new_str_7, "d_plus_f.pgm");
	write_pgm_blank_header(new_str_7, nx, ny);
	write_pgm_data(new_str_7, d_plus_f, nx, ny, bx, by);


	double** t1_plus_f;
	ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &t1_plus_f);
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++)
			t1_plus_f[i][j] = t1[i][j] + f[i][j];
	char new_str_8[200];
	strcpy(new_str_8, image_published_str);
	strcat(new_str_8, "t1_plus_f.pgm");
	write_pgm_blank_header(new_str_8, nx, ny);
	write_pgm_data(new_str_8, t1_plus_f, nx, ny, bx, by);


	double** t2_plus_f;
	ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &t2_plus_f);
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++)
			t2_plus_f[i][j] = t2[i][j] + f[i][j];
	char new_str_9[200];
	strcpy(new_str_9, image_published_str);
	strcat(new_str_9, "t2_plus_f.pgm");
	write_pgm_blank_header(new_str_9, nx, ny);
	write_pgm_data(new_str_9, t2_plus_f, nx, ny, bx, by);


	double** t2_plus_d;
	ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &t2_plus_d);
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++)
			t2_plus_d[i][j] = t2[i][j] + d[i][j];
	char new_str_10[200];
	strcpy(new_str_10, image_published_str);
	strcat(new_str_10, "t2_plus_d.pgm");
	write_pgm_blank_header(new_str_10, nx, ny);
	write_pgm_data(new_str_10, t2_plus_d, nx, ny, bx, by);


	double** t1_plus_d;
	ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &t1_plus_d);
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++)
			t1_plus_d[i][j] = t1[i][j] + d[i][j];
	char new_str_11[200];
	strcpy(new_str_11, image_published_str);
	strcat(new_str_11, "t1_plus_d.pgm");
	write_pgm_blank_header(new_str_11, nx, ny);
	write_pgm_data(new_str_11, t1_plus_d, nx, ny, bx, by);


	double** t2_plus_t1;
	ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &t2_plus_t1);
	for (i = bx; i < nx + bx; i++)
		for (j = by; j < ny + by; j++)
			t2_plus_t1[i][j] = t2[i][j] + t1[i][j];
	char new_str_12[200];
	strcpy(new_str_12, image_published_str);
	strcat(new_str_12, "t2_plus_t1.pgm");
	write_pgm_blank_header(new_str_12, nx, ny);
	write_pgm_data(new_str_12, t2_plus_t1, nx, ny, bx, by);
}


void median_filter(double** input_image, int nx, int ny, int bx, int by, double** output_image)
{
	double array[9];
	double temp;

	for (i = bx; i < nx + bx; i++)
	{
		for (j = by; j < ny + by; j++)
		{
			array[0] = input_image[i-1][j-1];
			array[1] = input_image[i-1][j];
			array[2] = input_image[i-1][j+1];
			array[3] = input_image[i][j-1];
			array[4] = input_image[i][j];
			array[5] = input_image[i][j+1];
			array[6] = input_image[i+1][j-1];
			array[7] = input_image[i+1][j];
			array[8] = input_image[i+1][j+1];

			temp = return_median_value(array);
			output_image[i][j] = temp;
		}
	}
}

double return_median_value (double array[])
{
	// sorting the array, and return the middle value (index=4) ~ median value
	double temp;
	for (int var = 0; var < 9; var++)
	{
		for (int var2 = var+1; var2 < 9; var2++)
		{
			if(array[var] >= array[var2])
			{
				temp = array[var];
				array[var] = array[var2];
				array[var2] = temp;
			}
		}
	}
	return array[4];
}


//=================================================================================================
//=================================================================================================

// multiple sequences (10 before + 1 higher average gray value + 10 after)
/*
 int main (int argc, char* argv[])
 {
 ground_truth_exists = true;
 draw = false;

 seg_type= (int)CV_;

 bx = by = 1;
 hx = hy = 1.0;
 sigma =   1.0;
 tau =     0.15;
 T =       1.0;
 T_incr =  1.0;
 epsilon = 0.1;
 nu =      0.0;
 mu =      1000.0;
 lambda11 = 10.0;
 lambda12 = 10.0;
 lambda21 = 5.0;
 lambda22 = 5.0;

 char image_dwi[MAXLINE];
 char image_flair[MAXLINE];
 char image_ot[MAXLINE];
 char temp[5];

 int scale = 10;
 int max_no_images = (scale*2) +1;
 // Higher Average Gray Value for the Ground Truth image
 int central[25] = {91,95,69,90,74,90,90,96,51,68,79,79,68,70,90,108,100,134,96,18,101,56,79,117,126};

 printf("---------------------------------------- DWI ------------------------------------------------ \n");
 for(int image_counter = 7 ; image_counter < 25 ; image_counter++)
 {
 int index = central[image_counter];
 printf(" ^^^^ image no: %d ^^^^ \n", image_counter+1);

 for (int i = index-scale; i <= index+scale; i++)
 {
 time_t mytime = time(0);
 char * time_str = ctime(&mytime);
 time_str[strlen(time_str)-1] = '\0';
 printf("Current Time : %s\n", time_str);

 if(i==index)
 printf("**(%d):", i);
 else
 printf("(%d):", i);

 strcpy(image_dwi, "/home/abouelen/eclipse-workspace-main/data_v5/");
 strcpy(image_ot, "/home/abouelen/eclipse-workspace-main/data_v5/");

 sprintf(temp, "%d", image_counter+1);
 strcat(image_dwi, temp);
 strcat(image_ot, temp);

 strcat(image_dwi, "/dwi/");
 strcat(image_ot, "/ot/");

 sprintf(temp, "%d", i);
 strcat(image_dwi, temp);
 strcat(image_ot, temp);

 strcat(image_dwi, ".pgm");
 strcat(image_ot, ".pgm");

 printf("%s \n", image_dwi);

 read_pgm_header(image_dwi,&position,&nx,&ny);
 ALLOC_MATRIX(2, nx+2*bx, ny+2*by, &f_c1, &f_c1_s);
 read_pgm_data(image_dwi,position,f_c1,nx,ny,bx,by);

 read_pgm_header(image_ot,&position,&nx,&ny);
 ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &f_ot);
 read_pgm_data(image_ot,position,f_ot,nx,ny,bx,by);

 ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &out);
 ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &f_post_seg);
 ALLOC_CUBIX(1, 2*nx+2*bx, ny+2*by, 3, &p6);
 ALLOC_MATRIX(4, nx+2*bx, ny+2*by, &phi1, &phi2, &phi_stored1, &phi_stored2);

 handleCompute();

 // free memory
 FREE_MATRIX(1, nx+2*bx, ny+2*by, f_ot);
 FREE_MATRIX(2, nx+2*bx, ny+2*by, f_c1, f_c1_s);                   // image channel 1

 FREE_MATRIX(1, nx+2*bx, ny+2*by, out);
 FREE_MATRIX(1, nx+2*bx, ny+2*by, f_post_seg);                   // post segmented image
 FREE_CUBIX(1, 2*nx+2*bx, ny+2*by, 3, p6);
 FREE_MATRIX(4, nx+2*bx, ny+2*by, phi1, phi2, phi_stored1, phi_stored2);
 printf("================================================================\n");
 }
 printf("average score for image no (%d) = %f \n", image_counter+1, average_score/max_no_images);
 average_score = 0;
 printf("\n");
 printf("\n");
 printf("\n");
 }

 printf("----------------------------------------------------------------------------------------------- \n");
 printf("----------------------------------------------------------------------------------------------- \n");
 printf("---------------------------------------- FLAIR ------------------------------------------------ \n");
 for(int image_counter = 0 ; image_counter < 25 ; image_counter++)
 {
 int index = central[image_counter];
 printf(" ^^^^ image no: %d ^^^^ \n", image_counter+1);
 for (int i = index-scale; i <= index+scale; i++)
 {
 time_t mytime = time(0);
 char * time_str = ctime(&mytime);
 time_str[strlen(time_str)-1] = '\0';
 printf("Current Time : %s\n", time_str);

 if(i==index)
 printf("**(%d):", i);
 else
 printf("(%d):", i);

 strcpy(image_flair, "/home/abouelen/eclipse-workspace-main/data_v5/");
 strcpy(image_ot, "/home/abouelen/eclipse-workspace-main/data_v5/");

 sprintf(temp, "%d", image_counter+		for(i = bx; i < nx + bx; i++)
 for(j = by; j < ny + by; j++)
 {
 //				printf("=[%d][%d] = %f, %f, %f \n", i,j, f_c1[i][j], f_c2[i][j], f_c3[i][j]);
 //				if(i%10==0 && j%10==0)
 //					printf("HOP\n");
 if(f_c2[i][j]!=0)
 printf("=[%d][%d] = %f, %f, %f \n", i,j, f_c1[i][j], f_c2[i][j], f_c3[i][j]);
 if(f_c3[i][j]!=0)
 printf("=[%d][%d] = %f, %f, %f \n", i,j, f_c1[i][j], f_c2[i][j], f_c3[i][j]);1);
 strcat(image_flair, temp);
 strcat(image_ot, temp);

 strcat(image_flair, "/flair/");
 strcat(image_ot, "/ot/");

 sprintf(temp, "%d", i);
 strcat(image_flair, temp);
 strcat(image_ot, temp);

 strcat(image_flair, ".pgm");
 strcat(image_ot, ".pgm");

 printf("%s \n", image_flair);

 read_pgm_header(image_flair,&position,&nx,&ny);
 ALLOC_MATRIX(2, nx+2*bx, ny+2*by, &f_c1, &f_c1_s);
 read_pgm_data(image_flair,position,f_c1,nx,ny,bx,by);

 read_pgm_header(image_ot,&position,&nx,&ny);
 ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &f_ot);
 read_pgm_data(image_ot,position,f_ot,nx,ny,bx,by);

 ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &out);
 ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &f_post_seg);
 ALLOC_CUBIX(1, 2*nx+2*bx, ny+2*by, 3, &p6);
 ALLOC_MATRIX(4, nx+2*bx, ny+2*by, &phi1, &phi2, &phi_stored1, &phi_stored2);

 handleCompute();

 // free memory
 FREE_MATRIX(1, nx+2*bx, ny+2*by, f_ot);
 FREE_MATRIX(2, nx+2*bx, ny+2*by, f_c1, f_c1_s);                   // image channel 1

 FREE_MATRIX(1, nx+2*bx, ny+2*by, out);
 FREE_MATRIX(1, nx+2*bx, ny+2*by, f_post_seg);                   // post segmented image
 FREE_CUBIX(1, 2*nx+2*bx, ny+2*by, 3, p6);
 FREE_MATRIX(4, nx+2*bx, ny+2*by, phi1, phi2, phi_stored1, phi_stored2);
 printf("================================================================\n");
 }
 printf("average score for image no (%d) = %f \n", image_counter+1, average_score/max_no_images);
 average_score = 0;
 printf("\n");
 printf("\n");
 printf("\n");
 }
 }
 */
/*
 // vectorized (Flair, DWI, T1) + 11 sequences (5 before + 1 + 5 after)
 int main (int argc, char* argv[])
 {
 ground_truth_exists = true;
 draw = false;

 bx = by = 1;
 hx = hy = 1.0;
 sigma =   1.0;
 tau =     0.15;
 T =       1.0;
 T_incr =  1.0;
 epsilon = 0.1;
 nu =      0.0;
 mu =      1000.0;
 seg_type= (int)CV_VEC;

 lambda11 = 10.0;
 lambda12 = 10.0;
 lambda21 = 5.0;
 lambda22 = 5.0;

 char image_dwi[MAXLINE];
 char image_flair[MAXLINE];
 char image_t1[MAXLINE];
 char image_ot[MAXLINE];
 char temp[5];

 int scale = 5;
 int max_no_images = (scale*2) +1;
 // Higher Average Gray Value for the Ground Truth image
 int central[25] = {91,95,69,90,74,90,90,96,51,68,79,79,68,70,90,108,100,134,96,18,101,56,79,117,126};

 clock_t start, end;
 double cpu_time_used;

 for(int image_counter = 0 ; image_counter < 25 ; image_counter++)
 {
 int index = central[image_counter];
 printf(" ^^^^ image no: %d ^^^^ \n", image_counter+1);
 time_t mytime = time(0);
 char * time_str = ctime(&mytime);
 time_str[strlen(time_str)-1] = '\0';
 printf("Current Time : %s\n", time_str);
 start = clock();

 for (int i = index-scale; i <= index+scale; i++)
 {
 if(i==index)
 printf("**(%d):", i);
 else
 printf("(%d):", i);

 strcpy(image_dwi, "/home/abouelen/eclipse-workspace-main/data_v5/");
 strcpy(image_flair, "/home/abouelen/eclipse-workspace-main/data_v5/");
 strcpy(image_t1, "/home/abouelen/eclipse-workspace-main/data_v5/");
 strcpy(image_ot, "/home/abouelen/eclipse-workspace-main/data_v5/");

 sprintf(temp, "%d", image_counter+1);
 strcat(image_dwi, temp);
 strcat(image_flair, temp);
 strcat(image_t1, temp);
 strcat(image_ot, temp);

 strcat(image_dwi, "/dwi/");
 strcat(image_flair, "/flair/");
 strcat(image_t1, "/t1/");
 strcat(image_ot, "/ot/");

 sprintf(temp, "%d", i);
 strcat(image_dwi, temp);
 strcat(image_flair, temp);
 strcat(image_t1, temp);
 strcat(image_ot, temp);

 strcat(image_dwi, ".pgm");
 strcat(image_flair, ".pgm");
 strcat(image_t1, ".pgm");
 strcat(image_ot, ".pgm");

 read_pgm_header(image_dwi,&position1,&nx,&ny);
 read_pgm_header(image_flair,&position2,&nx,&ny);
 read_pgm_header(image_t1,&position3,&nx,&ny);
 read_pgm_header(image_ot,&position,&nx,&ny);

 ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &f_ot);
 ALLOC_MATRIX(2, nx+2*bx, ny+2*by, &f_c1, &f_c1_s);
 ALLOC_MATRIX(2, nx+2*bx, ny+2*by, &f_c2, &f_c2_s);
 ALLOC_MATRIX(2, nx+2*bx, ny+2*by, &f_c3, &f_c3_s);
 ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &out);
 ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &f_post_seg);
 ALLOC_CUBIX(1, 2*nx+2*bx, ny+2*by, 3, &p6);
 ALLOC_MATRIX(4, nx+2*bx, ny+2*by, &phi1, &phi2, &phi_stored1, &phi_stored2);

 read_pgm_data(image_dwi,position1,f_c1,nx,ny,bx,by);
 read_pgm_data(image_flair,position2,f_c2,nx,ny,bx,by);
 read_pgm_data(image_t1,position3,f_c3,nx,ny,bx,by);
 read_pgm_data(image_ot,position,f_ot,nx,ny,bx,by);

 handleCompute();

 FREE_MATRIX(1, nx+2*bx, ny+2*by, f_ot);
 FREE_MATRIX(2, nx+2*bx, ny+2*by, f_c1, f_c1_s);
 FREE_MATRIX(2, nx+2*bx, ny+2*by, f_c2, f_c2_s);
 FREE_MATRIX(2, nx+2*bx, ny+2*by, f_c3, f_c3_s);
 FREE_MATRIX(1, nx+2*bx, ny+2*by, out);
 FREE_MATRIX(1, nx+2*bx, ny+2*by, f_post_seg);
 FREE_CUBIX(1, 2*nx+2*bx, ny+2*by, 3, p6);
 FREE_MATRIX(4, nx+2*bx, ny+2*by, phi1, phi2, phi_stored1, phi_stored2);
 printf("================================================================\n");
 }
 printf("average score for image no (%d) = %f \n", image_counter+1, average_score/max_no_images);
 average_score = 0;

 end = clock();
 cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
 printf("cpu_time_used = %3.1f minutes \n", cpu_time_used/60);

 printf("\n");
 printf("\n");
 }
 }
 */

/*
 // vectorized (Flair, DWI, T1, T2)
 int main (int argc, char* argv[])
 {
 ground_truth_exists = true;
 draw = false;
 seg_type= (int)CV_VEC;

 bx = by = 1;
 hx = hy = 1.0;
 sigma =   1.0;
 tau =     0.15;
 T =       1.0;
 T_incr =  1.0;
 epsilon = 0.1;
 nu =      0.0;
 mu =      1000.0;

 lambda11 = 10.0;
 lambda12 = 7.0;
 lambda21 = 5.0;
 lambda22 = 5.0;

 DIR* dir;
 DIR* dir2;
 struct dirent* ent;
 struct dirent* ent2;

 if ((dir = opendir ("/home/abouelen/eclipse-workspace-main/data_v4_higher_avg_OT/")) != NULL)
 {
 while((ent = readdir (dir)) != NULL)
 {
 if(!strcmp(ent->d_name,".") || !strcmp(ent->d_name,".."))
 continue;
 printf("Folder no (%s) \n", ent->d_name);

 strcpy(image, "/home/abouelen/eclipse-workspace-main/data_v4_higher_avg_OT/");
 strcat(image, ent->d_name);
 strcat(image, "/");

 strcpy(image_chan_dwi, image);
 strcpy(image_chan_flair, image);
 strcpy(image_chan_t1, image);
 strcpy(image_chan_t2, image);
 strcpy(image_OT, image);

 if ((dir2 = opendir (image)) != NULL)
 {
 while((ent2 = readdir (dir2)) != NULL)
 {
 if(!strcmp(ent2->d_name,".") || !strcmp(ent2->d_name,".."))
 continue;

 if(strncmp(ent2->d_name, "OT", 2) == 0)
 strcat(image_OT, ent2->d_name);
 else if(strncmp(ent2->d_name, "Flair", 5) == 0)
 strcat(image_chan_flair, ent2->d_name);
 else if(strncmp(ent2->d_name, "DWI", 3) == 0)
 strcat(image_chan_dwi, ent2->d_name);
 else if(strncmp(ent2->d_name, "T1", 2) == 0)
 strcat(image_chan_t1, ent2->d_name);
 else if(strncmp(ent2->d_name, "T2", 2) == 0)
 strcat(image_chan_t2, ent2->d_name);
 }

 read_pgm_header(image_chan_dwi,&position1,&nx,&ny);
 read_pgm_header(image_chan_flair,&position2,&nx,&ny);
 read_pgm_header(image_chan_t1,&position3,&nx,&ny);
 read_pgm_header(image_chan_t2,&position4,&nx,&ny);
 read_pgm_header(image_OT,&position,&nx,&ny);

 ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &f_ot);
 ALLOC_MATRIX(2, nx+2*bx, ny+2*by, &f_c1, &f_c1_s);
 ALLOC_MATRIX(2, nx+2*bx, ny+2*by, &f_c2, &f_c2_s);
 ALLOC_MATRIX(2, nx+2*bx, ny+2*by, &f_c3, &f_c3_s);
 ALLOC_MATRIX(2, nx+2*bx, ny+2*by, &f_c4, &f_c4_s);
 ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &out);
 ALLOC_MATRIX(1, nx+2*bx, ny+2*by, &f_post_seg);
 ALLOC_CUBIX(1, 2*nx+2*bx, ny+2*by, 3, &p6);
 ALLOC_MATRIX(4, nx+2*bx, ny+2*by, &phi1, &phi2, &phi_stored1, &phi_stored2);

 read_pgm_data(image_chan_dwi,position1,f_c1,nx,ny,bx,by);
 read_pgm_data(image_chan_flair,position2,f_c2,nx,ny,bx,by);
 read_pgm_data(image_chan_t1,position3,f_c3,nx,ny,bx,by);
 read_pgm_data(image_chan_t2,position4,f_c4,nx,ny,bx,by);
 read_pgm_data(image_OT,position,f_ot,nx,ny,bx,by);

 strcpy(out_name, image);
 strcat(out_name, "out_vec_");
 strcat(out_name, ent->d_name);

 handleCompute();

 FREE_MATRIX(1, nx+2*bx, ny+2*by, f_ot);
 FREE_MATRIX(2, nx+2*bx, ny+2*by, f_c1, f_c1_s);
 FREE_MATRIX(2, nx+2*bx, ny+2*by, f_c2, f_c2_s);
 FREE_MATRIX(2, nx+2*bx, ny+2*by, f_c3, f_c3_s);
 FREE_MATRIX(2, nx+2*bx, ny+2*by, f_c4, f_c4_s);
 FREE_MATRIX(1, nx+2*bx, ny+2*by, out);
 FREE_MATRIX(1, nx+2*bx, ny+2*by, f_post_seg);
 FREE_CUBIX(1, 2*nx+2*bx, ny+2*by, 3, p6);
 FREE_MATRIX(4, nx+2*bx, ny+2*by, phi1, phi2, phi_stored1, phi_stored2);

 printf("\n\n");
 }
 closedir(dir2);
 }
 }
 closedir(dir);
 }
 */

// user input
int main(int argc, char* argv[]) {

	draw = true;
	publish_segmented_image = true;

	long position_1;
	long position_2;
	long position_3;
	long position_4;
	long positionOT;

	// TODO
	bx = by = 1;
	hx = hy = 1.0;
	sigma = 1.0;		// smoothing the image
	tau = 0.1;
	T = 1.0;			// not used
	T_incr = 1.0;		// not used
	epsilon = 0.1;		// used for delta function
	nu = 0;				// not used
	mu = 10.0;			// weight length constraint

	lambda11 = 1.0;
	lambda12 = 1.0;
	lambda21 = 1.0;
	lambda22 = 1.0;


	/*
	// understand the arguments (recognize the type of the image) !not important at the moment
	// --------------------------------------------------------------------
//	printf("No of arguments = %d\n", argc);
	for (int i = 1; i < argc; i++) // start with counter 1 to skip the RUN name
	{
//		printf("argv[%d] = %s\n", i, argv[i]);
		char * u;
		int backSlashCharacter = '\/';
		u = strrchr(argv[i], backSlashCharacter);
		int index = u - argv[i];
		char imageName[100];
		int intTemp = 0;
		while (intTemp < index + 1) {
			imageName[intTemp] = argv[i][intTemp];
			intTemp++;
		}
		imageName[intTemp] = '\0';
		char* contents_chopped = u + 1;
//		printf("subTemp = %s \n", contents_chopped);
		int underscore = '_';
		u = strchr(contents_chopped, underscore);

		int intTemp2 = u - contents_chopped;
		char type[10];
		intTemp = 0;
		while (intTemp < intTemp2) {
			type[intTemp] = contents_chopped[intTemp];
			intTemp++;
		}
		type[intTemp] = '\0';
		printf("type = %s \n", type);
	}
*/
	// --------------------------------------------------------------------

//	argc = 1 : input error
//	argc = 2 : input error
//	argc = 3 : base + one image + OT		CV
//	argc = 4 : base + two images + OT		CV_VEC_2
//	argc = 5 : base + three images + OT 	CV_VEC_3
//	argc = 6 : base + four images + OT  	CV_VEC_4

	printf("argc = %d \n", argc);
	for (int var = 0; var < argc; var++)
		printf("(%d) %s\n", var, argv[var]);

		if (argc == 1 || argc == 2)
		{
			printf("Input Error \n");
			return 0;
		}
		if (argc == 3)
		{
			seg_type = (int) CV_;
			strcpy(image_chan_1_str, argv[1]);

			// determine dimensions
			if (strcmp(get_extension(image_chan_1_str), ".pgm") == 0)
				read_pgm_header(image_chan_1_str, &position_1, &nx, &ny);

			if (OT_exists)
			{
				strcpy(image_OT_str, argv[2]);
				if (strcmp(get_extension(image_OT_str), ".pgm") == 0)
					read_pgm_header(image_OT_str, &positionOT, &nx, &ny);
				ALLOC_MATRIX(1, nx + 2 * bx, ny + 2 * by, &f_ot);
			}
		}
		else if(argc == 4)
		{
			seg_type = (int) CV_VEC_2;

			strcpy(image_chan_1_str, argv[1]);
			if (strcmp(get_extension(image_chan_1_str), ".pgm") == 0)
				read_pgm_header(image_chan_1_str, &position_1, &nx, &ny);

			strcpy(image_chan_2_str, argv[2]);
			if (strcmp(get_extension(image_chan_2_str), ".pgm") == 0)
				read_pgm_header(image_chan_2_str, &position_2, &nx, &ny);

			if (OT_exists) {
				strcpy(image_OT_str, argv[3]);
				if (strcmp(get_extension(image_OT_str), ".pgm") == 0)
					read_pgm_header(image_OT_str, &positionOT, &nx, &ny);
				ALLOC_MATRIX(1, nx + 2 * bx, ny + 2 * by, &f_ot);
			}
		}
		else if(argc == 5)
		{
			seg_type = (int) CV_VEC_3;
			strcpy(image_chan_1_str, argv[1]);
			if (strcmp(get_extension(image_chan_1_str), ".pgm") == 0)
				read_pgm_header(image_chan_1_str, &position_1, &nx, &ny);
			strcpy(image_chan_2_str, argv[2]);
			if (strcmp(get_extension(image_chan_2_str), ".pgm") == 0)
				read_pgm_header(image_chan_2_str, &position_2, &nx, &ny);
			strcpy(image_chan_3_str, argv[3]);
			if (strcmp(get_extension(image_chan_3_str), ".pgm") == 0)
				read_pgm_header(image_chan_3_str, &position_3, &nx, &ny);

			if (OT_exists) {
				strcpy(image_OT_str, argv[4]);
				if (strcmp(get_extension(image_OT_str), ".pgm") == 0)
					read_pgm_header(image_OT_str, &positionOT, &nx, &ny);
				ALLOC_MATRIX(1, nx + 2 * bx, ny + 2 * by, &f_ot);

			}
		}
		else if(argc == 6)
		{
			seg_type = (int) CV_VEC_4;

			strcpy(image_chan_1_str, argv[1]);
			if (strcmp(get_extension(image_chan_1_str), ".pgm") == 0)
				read_pgm_header(image_chan_1_str, &position_1, &nx, &ny);

			strcpy(image_chan_2_str, argv[2]);
			if (strcmp(get_extension(image_chan_2_str), ".pgm") == 0)
				read_pgm_header(image_chan_2_str, &position_2, &nx, &ny);

			strcpy(image_chan_3_str, argv[3]);
			if (strcmp(get_extension(image_chan_3_str), ".pgm") == 0)
				read_pgm_header(image_chan_3_str, &position_3, &nx, &ny);

			strcpy(image_chan_4_str, argv[4]);
			if (strcmp(get_extension(image_chan_4_str), ".pgm") == 0)
				read_pgm_header(image_chan_4_str, &position_4, &nx, &ny);

			if (OT_exists) {
				strcpy(image_OT_str, argv[5]);
				if (strcmp(get_extension(image_OT_str), ".pgm") == 0)
					read_pgm_header(image_OT_str, &positionOT, &nx, &ny);
				ALLOC_MATRIX(1, nx + 2 * bx, ny + 2 * by, &f_ot);

			}
		}


		// allocate memory
		ALLOC_MATRIX(2, nx + 2 * bx, ny + 2 * by, &f_c1, &f_c1_s); // image channel 1
		ALLOC_MATRIX(2, nx + 2 * bx, ny + 2 * by, &f_c2, &f_c2_s); // image channel 2
		ALLOC_MATRIX(2, nx + 2 * bx, ny + 2 * by, &f_c3, &f_c3_s); // image channel 3
		ALLOC_MATRIX(2, nx + 2 * bx, ny + 2 * by, &f_c4, &f_c4_s); // image channel 3
		ALLOC_MATRIX(2, nx + 2 * bx, ny + 2 * by, &f2, &f2_s);       // second image_str
		ALLOC_MATRIX(1, nx + 2 * bx, ny + 2 * by, &f_work);          // working copy
		ALLOC_MATRIX(2, nx + 2 * bx, ny + 2 * by, &u, &v);           // optical flow
		ALLOC_MATRIX(2, nx + 2 * bx, ny + 2 * by, &u_s, &v_s); // smoothed optical flow
		ALLOC_MATRIX(1, nx + 2 * bx, ny + 2 * by, &out);                   // output
		ALLOC_MATRIX(2, nx + 2 * bx, ny + 2 * by, &phi, &phi_stored); // level set function
		ALLOC_MATRIX(1, nx + 2 * bx, ny + 2 * by, &f_post_seg); // post segmented image
		ALLOC_CUBIX(1, 2 * nx + 2 * bx, ny + 2 * by, 3, &p6);
		ALLOC_MATRIX(4, nx + 2 * bx, ny + 2 * by, &phi1, &phi2, &phi_stored1,&phi_stored2);

		pixels = (GLubyte *) malloc(2 * (nx + 6) * (ny + 6) * 3 * sizeof(GLubyte));


		// --------------------------------------------------------- preprocess for the extracted image
		char * image_title_with_backslash;
		int backSlashCharacter = '\/';
		image_title_with_backslash = strrchr(image_chan_1_str, backSlashCharacter);
		int index = image_title_with_backslash - image_chan_1_str;
		char folder_path[100];
		int g = 0;
		while (g < index + 1) {
			folder_path[g] = image_chan_1_str[g];
			g++;
		}
		folder_path[g] = '\0';
		char* image_title_chopped = image_title_with_backslash + 1;
		strcpy(image_published_str, folder_path);
		// TODO
		strcpy(image_title_str, image_title_chopped);
		// END ------------------------------------------------------preprocess for the extracted image


		if (argc == 3)
		{
			// read the data
			if (strcmp(get_extension(image_chan_1_str), ".pgm") == 0) {
				read_pgm_data(image_chan_1_str, position_1, f_c1, nx, ny, bx, by);
				// copy to other images channels for visualisation
				copy_matrix_2d(f_c1, f_c2, nx, ny, bx, by);
				copy_matrix_2d(f_c1, f_c3, nx, ny, bx, by);
			}
			if (OT_exists) {
				read_pgm_data(image_OT_str, positionOT, f_ot, nx, ny, bx, by);
			}

			// prefix for the published image
			strcat(image_published_str, "out_CV_VEC_1_");
		}
		else if(argc == 4)
		{
			// read the data
			if (strcmp(get_extension(image_chan_1_str), ".pgm") == 0)
				read_pgm_data(image_chan_1_str, position_1, f_c1, nx, ny, bx, by);
			if (strcmp(get_extension(image_chan_2_str), ".pgm") == 0)
				read_pgm_data(image_chan_2_str, position_2, f_c2, nx, ny, bx, by);

			// Ground Truth
			if (OT_exists)
				read_pgm_data(image_OT_str, positionOT, f_ot, nx, ny, bx, by);

			// prefix for the published image
			strcat(image_published_str, "out_CV_VEC_2_");
		}
		else if(argc == 5)
		{
			// read the data
			if (strcmp(get_extension(image_chan_1_str), ".pgm") == 0)
				read_pgm_data(image_chan_1_str, position_1, f_c1, nx, ny, bx, by);
			if (strcmp(get_extension(image_chan_2_str), ".pgm") == 0)
				read_pgm_data(image_chan_2_str, position_2, f_c2, nx, ny, bx, by);
			if (strcmp(get_extension(image_chan_3_str), ".pgm") == 0)
				read_pgm_data(image_chan_3_str, position_3, f_c3, nx, ny, bx, by);

			// Ground Truth
			if (OT_exists)
				read_pgm_data(image_OT_str, positionOT, f_ot, nx, ny, bx, by);

			// prefix for the published image
			strcat(image_published_str, "out_CV_VEC_3_");
		}
		else if(argc == 6)
		{
			// read the data
			if (strcmp(get_extension(image_chan_1_str), ".pgm") == 0)
				read_pgm_data(image_chan_1_str, position_1, f_c1, nx, ny, bx, by);
			if (strcmp(get_extension(image_chan_2_str), ".pgm") == 0)
				read_pgm_data(image_chan_2_str, position_2, f_c2, nx, ny, bx, by);
			if (strcmp(get_extension(image_chan_3_str), ".pgm") == 0)
				read_pgm_data(image_chan_3_str, position_3, f_c3, nx, ny, bx, by);
			if (strcmp(get_extension(image_chan_4_str), ".pgm") == 0)
				read_pgm_data(image_chan_4_str, position_4, f_c4, nx, ny, bx, by);

			// Ground Truth
			if (OT_exists)
				read_pgm_data(image_OT_str, positionOT, f_ot, nx, ny, bx, by);

			// prefix for the published image
			strcat(image_published_str, "out_CV_VEC_4_");
		}
		// for the published image
//		strcat(image_published_str, image_title_chopped);


		// ---------- M A I N   L O O P ---------------------------------------------
		strcpy(title, "TEST CODE FOR SEGMENTATION ");
		strcat(title, get_name(image_chan_1_str));
		glutInit(&argc, argv);
		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
		glutInitWindowSize((int) round(2 * nx), (int) round(ny));
		glutCreateWindow(title);
		glutDisplayFunc(handleDraw);
		glutIdleFunc(handleComputeNothing);
		glutKeyboardFunc(handleKeyboard);
		glutSpecialFunc(handleKeyboardspecial);
		glutMouseFunc(handleMouse);
		showParams();
		glutMainLoop();
		handleCompute();

		// free memory
		if (OT_exists)
			FREE_MATRIX(1, nx + 2 * bx, ny + 2 * by, f_ot);
		FREE_MATRIX(2, nx + 2 * bx, ny + 2 * by, f_c1, f_c1_s);   // image channel 1
		FREE_MATRIX(2, nx + 2 * bx, ny + 2 * by, f_c2, f_c2_s);   // image channel 2
		FREE_MATRIX(2, nx + 2 * bx, ny + 2 * by, f_c3, f_c3_s);   // image channel 3
		FREE_MATRIX(2, nx + 2 * bx, ny + 2 * by, f_c4, f_c4_s);   // image channel 4
		FREE_MATRIX(2, nx + 2 * bx, ny + 2 * by, f2, f2_s);	         // second image_str
		FREE_MATRIX(1, nx + 2 * bx, ny + 2 * by, f_work);            // working copy
		FREE_MATRIX(2, nx + 2 * bx, ny + 2 * by, u, v);              // optical flow
		FREE_MATRIX(2, nx + 2 * bx, ny + 2 * by, u_s, v_s); // smoothed optical flow
		FREE_MATRIX(1, nx + 2 * bx, ny + 2 * by, out);	                   // output
		FREE_MATRIX(1, nx + 2 * bx, ny + 2 * by, f_post_seg); // post segmented image_str
		FREE_MATRIX(2, nx + 2 * bx, ny + 2 * by, phi, phi_stored); // level set function
		FREE_CUBIX(1, 2 * nx + 2 * bx, ny + 2 * by, 4, p6);
		FREE_MATRIX(4, nx + 2 * bx, ny + 2 * by, phi1, phi2, phi_stored1,phi_stored2);
		free(pixels);

		printf("\n\n");
}

