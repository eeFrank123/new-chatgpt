#pragma once
#define ASCENDC_TPL_ARGS_DECL(...)
#define ASCENDC_TPL_SEL(...)
#define ASCENDC_TPL_SEL_PARAM(context, dtype, io_mode) \
    (context)->SetTilingKey(static_cast<uint64_t>(dtype) | (static_cast<uint64_t>(io_mode) << 8U))
