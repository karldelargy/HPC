#ifndef EQUATIONS_H
#define EQUATIONS_H

#include "dataStructiars.h"
#include "global_values.h"
/** \fn landNeighboursCells
 * \brief  brief description of what the fucntion does
 *
 *
 * A more detailed description could go here
 *
 *
 *  \param gameland talk about what gameland does
 *  \param i talk about what i does
 *  \param j talk about what j does
 *  \return n.
 */

double landNeighboursCells(const struct_matrix *gameLand, int i, int j);

/** \fn haresNeighboursCells
 * \brief  brief description of what the fucntion does
 *
 *
 * A more detailed description could go here
 *
 *
 *  \param gameland talk about what gameland does
 *  \param i talk about what i does
 *  \param j talk about what j does
 *  \return Void.
 */

double haresNeighboursCells(const struct_matrix *gameLand, int i, int j);

/** \fn haresNewValue
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

double haresNewValue(struct_matrix *gameLand, int i, int j);

/** \fn pumasNeighboursCells
 * \brief  brief description of what the fucntion does
 *
 * A more detailed description could go here
 *
 *
 *  \param gameland talk about what gameland does
 *  \param i talk about what i does
 *  \param j talk about what j does
 */

double pumasNeighboursCells(const struct_matrix *gameLand, int i, int j);


/** \fn pumasNewValue
 * \brief  brief description of what the fucntion does
 * \fn pumasNewValue
 *
 * A more detailed description could go here
 *
 *
 *  \param gameland talk about what gameland does
 *  \param i talk about what i does
 *  \param j talk about what j does
 *  \return newValue.
 */

double pumasNewValue(struct_matrix *gameLand, int i, int j);

/** \fn mainLoop
 * \brief  brief description of what the fucntion does
 *
 *
 * A more detailed description could go here
 *
 *
 *  \param gameland talk about what gameland does
 *  \param newGameLand
 *  \return Void.
 */

void mainLoop(struct_matrix *gameLand, struct_matrix *newGameLand,  double *totalHares, double *totalPumas);

#endif
