#pragma once
#include "core/Tensor.h"
#include <vector>

class Embedding {
public:
    Embedding(int vocab_size, int d_model);
    Tensor forward(const std::vector<int>& token_ids);
    void zero_grad();

    Tensor& weights();  // for training/inspection

private:
    Tensor embedding_table;  // [vocab_size, d_model]
};
