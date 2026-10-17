// TilingKey模板定义的头文件
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(SafetyBeltViolationGeometry,
    ASCENDC_TPL_DATATYPE_DECL(DT_PERSON_BOXES, C_DT_FLOAT16, C_DT_FLOAT),
    // bits: threshold present, threshold stored as half, output stored as half.
    ASCENDC_TPL_UINT_DECL(IO_MODE, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 0, 1, 3, 4, 5, 7),
    // 0: general metadata. 1..33: B in
    // {1,32,33,256,1024,4096,64,128,129,257,512},
    // ratio in {0.8f,0.9f,1.0f}, with the corresponding fixed core partition.
    // 34..44: the same eleven shapes/partitions, with any legal float32 ratio
    // copied from tiling GM beside the actual input coordinates.
    ASCENDC_TPL_UINT_DECL(STATIC_CASE, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_DATATYPE_SEL(DT_PERSON_BOXES, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(IO_MODE, ASCENDC_TPL_UI_LIST, 0, 1),
        ASCENDC_TPL_UINT_SEL(STATIC_CASE, ASCENDC_TPL_UI_LIST,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_DATATYPE_SEL(DT_PERSON_BOXES, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(IO_MODE, ASCENDC_TPL_UI_LIST, 0, 1, 3, 4, 5, 7),
        ASCENDC_TPL_UINT_SEL(STATIC_CASE, ASCENDC_TPL_UI_LIST,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44),
    ),
);
