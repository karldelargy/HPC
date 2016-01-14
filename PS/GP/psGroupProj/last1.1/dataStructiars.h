#ifndef DATA_STRUCTIARS_H
#define DATA_STRUCTIARS_H

#include <stdio.h>
#include "global_values.h"

/** \enum enum_area
    \brief Enum.

    An enum where land is set to have a value of 1 and water has a value of zero.
*/

enum enum_area{LAND, WATER};

/*! \var typedef struct cell
    \typedef struct cell .

    Details.
*/

/*! \struct cell
    \A struct.

    We are making our cells so that they contain 3 things, whether they are land or water
    , the density of pumas and the density of hares.
*/

typedef struct cell {
  enum enum_area area;
  double pumas;
  double hares;
} struct_cell;

/*! \var typedef struct matrix
    \typedef struct matrix .

    Details.
*/

/*! \struct matrix
    \A struct.

    A more detailed class description.
*/

typedef struct matrix {
  /*\{*/
  int x; /**< 1D */
  int y; /**< 2D */
  struct_cell **map; /**< 3D */
  /*\}*/
} struct_matrix;

/** \fn dynamic_alloc_map
 * \brief  brief description of what the fucntion does
 * \fn dynamic_alloc_map
 *
 *
 *
 *
 *  \param x dimesnions of matrix
 *  \param y dimesnions of matrix
 *  \return map.
 */
struct_cell **dynamic_alloc_map(int x, int y);

/** * \fn free_map
 *\brief  brief description of what the fucntion does
 *
 *
 *
 *
 *  \param gameland talk about what gameland does
 *  \return Void.
 */
void free_map(struct_matrix *gameLand);

/** \fn init_map
 * \brief  brief description of what the fucntion does
 * \fn init_map
 *
 *
 * This one is quite big so I think we should explain this one
 * A little bit more
 *
 *  \param gameland talk about what gameland does
 *  \return Void.
 */

void init_map(FILE *fp, struct_matrix *gameLand);

#endif /* DATA_STRUCTIARS */

