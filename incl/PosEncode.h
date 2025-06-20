#pragma once
#include "core/Tensor.h"

class PositionalEncoding {
public:
    PositionalEncoding(int max_len, int d_model);
    Tensor get_encoding(int seq_len) const;

private:
    Tensor pe;  // [max_len, d_model]
};