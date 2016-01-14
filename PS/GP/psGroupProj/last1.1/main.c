/** \file doxygen.c
 *  \brief Puma and hare density
 *
 *  This contains code for determining the number of hares and
 *  Pumas in a grid. There will be water present on the grid, and
 *  in water squares, no important animals live.
 *
 *  \author Karl Delargy (kdelargy)
 *  \author Brian Flynn (bflynn)
 *  \author Killian Tattan (ktattan)
 *  \author Angelos Stylianides (astyliandes)
 *  \bug No known bugs.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "dataStructiars.h"
#include "equations.h"
#include "printFunctions.h"

#include "global_values.h"

double dt = 0.4;/**< change in time*/
double r = 0.08; /**< birthrate of hares */
double a = 0.04; /**< the predation rate of pumas */
double b = 0.02; /**< birth rate of hares per hare eaten */
double m = 0.06; /**< the puma morality rate */
double k = 0.2; /**< the diffusion rates for hares */
double l = 0.2;/**< diffusion rate for pumas */
double T = 5.0; /**< Time step */
double crit_hares_lower = 0.1;/**< Lower critical limit for pumas */
double crit_hares_upper = 5.0;/**< Upper critical limit for pumas */
double crit_pumas_lower = 0.1;/**< Lower critical limit for pumas */
double crit_pumas_upper = 5.0;/**< Upper critical limit for pumas */
double land_number = 0.0;
int grid_number = 0;

/**
*setting the colour scales for the output ppm files
*/
/*double min_colour = 70.0;
double scale_pumas = 37.2;//(256.0 - min_colour)/crit_pumas_upper;
double scale_hares = 37.2;//(256.0 - min_colour)/crit_hares_upper;*/

/** \fn Main
 * \brief  brief description of what the fucntion does
 *
 *
 * A more detailed description could go here
 *
 *
 *
 *
 */

int main(int argc, char **argv) {

  char *inputFile = NULL;
  FILE *fp = NULL;
  FILE *fpAvgLand = NULL;
  FILE *fpAvgGrid = NULL;
  struct_matrix *gameLand;
	struct_matrix *newGameLand;
  struct_cell **swap;

  int i, j;
  double t;
  double totalHares = 0.0;
  double avgHaresLand = 0.0;
  double totalPumas = 0.0;
  double avgPumasLand = 0.0;
  double avgHaresGrid = 0.0;
  double avgPumasGrid = 0.0;

  srand(time(NULL));

  //Check if argv 2 exist
  if (argc != 2)
  {
    fprintf(stderr, "No input file\n");
    return -1;
  }

  //get input file name
  inputFile = strdup(argv[1]);
  if (inputFile == NULL)
  {
    fprintf(stderr, "\\> function srtdup fail to allocate space and return \
            NULL.\n");
    return -1;
  }

  //open the input file
  fp = fopen(inputFile, "r+");
  if (fp == NULL)
  {
    fprintf(stderr, "\\> function fopen fail to open the file: %s\n", \
            inputFile);
    return -1;
  }

  gameLand = (struct_matrix*) malloc(sizeof(struct_matrix));
  //initialize map
  init_map(fp, gameLand);

  //free file ponter
  fclose(fp);

  //free file name store array
  free(inputFile);

  fpAvgLand=fopen("averageLand.txt", "w");
  if (fpAvgLand == NULL) {
    fprintf(stderr, "file append fail\n");
    return -1;
  }

  fpAvgGrid=fopen("averageGrid.txt", "w");
  if (fpAvgGrid == NULL) {
    fprintf(stderr, "file append fail\n");
    return -1;
  }

  //Print Header
  fprintf(fpAvgLand, "    Time    Average Hare Density    Average Puma Density\n");
  fprintf(fpAvgGrid, "    Time    Average Hare Density    Average Puma Density\n");

  newGameLand = (struct_matrix*) malloc(sizeof(struct_matrix));
  newGameLand->x = gameLand->x;
  newGameLand->y = gameLand->y;
  newGameLand->map = dynamic_alloc_map(newGameLand->x, newGameLand->y);

  for (i = 0; i < newGameLand->x; i++)
  {
    newGameLand->map[i][0].area = \
    newGameLand->map[i][newGameLand->y - 1].area = WATER;
    newGameLand->map[i][0].hares = \
    newGameLand->map[i][newGameLand->y - 1].hares = \
    newGameLand->map[i][0].pumas = \
    newGameLand->map[i][newGameLand->y - 1].pumas = 0.0;
  }

  for (j = 0; j < newGameLand->y; j++)
  {
    newGameLand->map[0][j].area = \
    newGameLand->map[newGameLand->x - 1][j].area = WATER;
    newGameLand->map[0][j].hares = \
    newGameLand->map[newGameLand->x - 1][j].hares = \
    newGameLand->map[0][j].pumas = \
    newGameLand->map[newGameLand->x - 1][j].pumas = 0.0;
  }

// calculating the number of grid points that are land
  for (i = 1; i < gameLand->x; i++) {
    for (j = 1; j < gameLand->y; j++) {
        if (gameLand->map[i][j].area == LAND) {
          land_number++;
        }
    }
}

grid_number = (gameLand->x - 2)*(gameLand->y - 2);
//  printMap(gameLand, "rawMap.txt");

//  printMapToImg(gameLand, "theMap");
//  printMap(gameLand, "theMap");

	for (t = 0.0;  t < 500.0; t += dt) {
    totalHares = 0.0;
    totalPumas = 0.0;
		mainLoop(gameLand, newGameLand, &totalHares, &totalPumas);
//	  printHares(gameLand);
//	  printPumas(gameLand);
    if(fmod(t,T)<dt){
 		   printPPM(gameLand);
       printHares(gameLand);
       printPumas(gameLand);

       // Calculate the average densities and output the result
       // average densities over land
       avgHaresLand = totalHares/((double)land_number);
       avgPumasLand = totalPumas/((double)land_number);

       // average densities over entire grid
       avgHaresGrid = totalHares/((double)grid_number);
       avgPumasGrid = totalPumas/((double)grid_number);

       printLandAvg(fpAvgLand, t, avgHaresLand, avgPumasLand);
       printGridAvg(fpAvgGrid, t, avgHaresGrid, avgPumasGrid);

    }

		//swap the arrays
    swap = gameLand->map;
    gameLand->map = newGameLand->map;
    newGameLand->map = swap;

//    for (i = 0; i < gameLand->x; i++) {
//      memcpy(gameLand->map[i], newGameLand->map[i], sizeof(struct_cell) * gameLand->y);
//    }

//	  print_pumas(gameLand);
//	  printf("------------------------------------------------------------\n");
//	  printPPM(gameLand);
//	  printHares(gameLand);
//	  printPumas(gameLand);
	}

  //free programme matrix
  free_map(gameLand);
  free_map(newGameLand);
  //TODO chech that
//  free(gameLand);

  fclose(fpAvgLand);
  fclose(fpAvgGrid);
  printf("Reach the end\n");
  return 0;
}
