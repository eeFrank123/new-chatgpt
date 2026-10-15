// TilingKey模板定义的头文件
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(SafetyBeltViolationGeometry,
    ASCENDC_TPL_DATATYPE_DECL(DT_PERSON_BOXES, C_DT_FLOAT16, C_DT_FLOAT),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_DATATYPE_SEL(DT_PERSON_BOXES, C_DT_FLOAT16, C_DT_FLOAT),
    ),
);
