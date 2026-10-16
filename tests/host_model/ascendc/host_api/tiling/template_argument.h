#pragma once
#define ASCENDC_TPL_ARGS_DECL(...)
#define ASCENDC_TPL_SEL(...)
#define ASCENDC_TPL_SEL_PARAM(context, dtype, io_mode, static_case) \
    (context)->SetTilingKey(static_cast<uint64_t>(dtype) | \
                           (static_cast<uint64_t>(io_mode) << 8U) | \
                           (static_cast<uint64_t>(static_case) << 16U))
