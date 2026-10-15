// TilingKey模板定义的头文件
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(SafetyBeltViolationGeometry,
    ASCENDC_TPL_DATATYPE_DECL(DT_PERSON_BOXES, C_DT_FLOAT16, C_DT_FLOAT),
    // bits: threshold present, threshold stored as half, output stored as half.
    ASCENDC_TPL_UINT_DECL(IO_MODE, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, 0, 1, 3, 4, 5, 7),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_DATATYPE_SEL(DT_PERSON_BOXES, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(IO_MODE, ASCENDC_TPL_UI_LIST, 0, 1),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_DATATYPE_SEL(DT_PERSON_BOXES, C_DT_FLOAT16),
        ASCENDC_TPL_UINT_SEL(IO_MODE, ASCENDC_TPL_UI_LIST, 0, 1, 3, 4, 5, 7),
    ),
);
