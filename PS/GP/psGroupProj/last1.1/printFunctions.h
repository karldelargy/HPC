#ifndef PRINT_FUNCTIONS_H
#define PRINT_FUNCTIONS_H


#include "dataStructiars.h"

/** \fn printPPM
 * \brief  brief description of what the fucntion does
 *
 * A more detailed description could go here
 *
 *
 *  \param gameland talk about what gameland does
 *  \return Void.
 */

int increace(float one, float two);

void printPPM(struct_matrix *gameLand);

/** \fn printMapToImg
 * \brief  brief description of what the fucntion does
 *
 *
 * A more detailed description could go here
 *
 *
 *  \param gameland talk about what gameland does
 *  \param i talk about what i does
 *  \param j talk about what j does
 *  \return newValue.
 */

void printMapToImg(struct_matrix *gameLand, char initFileName[30]);

void printLandAvg(FILE *fp, double t, double avgHaresLand, double avgPumasLand);

void printLandAvg(FILE *fp, double t, double avgHaresGrid, double avgPumasGrid);
#endif
