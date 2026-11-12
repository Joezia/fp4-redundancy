/*
** Copyright (C) 1999-2020 Erik de Castro Lopo <erikd@mega-nerd.com>
**
** This program is free software; you can redistribute it and/or modify
** it under the terms of the GNU Lesser General Public License as published by
** the Free Software Foundation; either version 2.1 of the License, or
** (at your option) any later version.
**
** This program is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
** GNU Lesser General Public License for more details.
**
** You should have received a copy of the GNU Lesser General Public License
** along with this program; if not, write to the Free Software
** Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
*/

// MODIFIED 2026-11-09: standalone IMA state and exact external compensation.
#pragma once
#include "common.hpp"
namespace predictive {
static int ima_indx_adjust [16] =
{	-1, -1, -1, -1,		/* +0 - +3, decrease the step size */
	+2, +4, +6, +8,		/* +4 - +7, increase the step size */
	-1, -1, -1, -1,		/* -0 - -3, decrease the step size */
	+2, +4, +6, +8,		/* -4 - -7, increase the step size */
} ;

static int ima_step_size [89] =
{	7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
	50, 55, 60, 66, 73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230,
	253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658, 724, 796, 876, 963,
	1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327,
	3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442,
	11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794,
	32767
} ;


// libsndfile 1.2.2 IMA/DVI loop, adapted to a component-local buffer.
// Keep the original quantized reconstruction feedback. A separate exact
// compensation stream restores the input codeword and is never fed into IMA.
struct Ima {
    int value=0,index=0;
    int decode(unsigned code) {
        int step=ima_step_size[index],difference=step>>3;
        if(code&1)difference+=step>>2;
        if(code&2)difference+=step>>1;
        if(code&4)difference+=step;
        value=std::clamp(value+((code&8)?-difference:difference),-32768,32767);
        index=std::clamp(index+ima_indx_adjust[code],0,88);
        return value;
    }
    unsigned encode(int sample) {
        int difference=sample-value,step=ima_step_size[index];unsigned code=0;
        if(difference<0){code=8;difference=-difference;}
        for(unsigned flag=4;flag;flag>>=1){if(difference>=step){code|=flag;difference-=step;}step>>=1;}
        decode(code);return code;
    }
};
}
