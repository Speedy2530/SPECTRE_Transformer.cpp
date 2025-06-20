#pragma once

#include <../third-party/Eigen/Dense>
#include <memory>
#include <vector>
#include <functional>
#include <string>
#include <iostream>

class Tensor;

// Used to store computation graph info for autograd
struct GradFn {
    std::string op_type;
    std::vector<std::shared_ptr<Tensor>> parents;
    std::function<void(const std::shared_ptr<Tensor>&)> backward;
};

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    // ===== Core Data =====
    Eigen::MatrixXf data;
    Eigen::MatrixXf grad;
    bool requires_grad = false;
    std::shared_ptr<GradFn> grad_fn = nullptr;

    // Track shape if needed
    int batch_size = 1;
    int seq_len = 1;
    int hidden_dim = 1;


    // ===== Constructors =====
    // Shape constructor
    Tensor(int rows, int cols, bool requires_grad = false);
    // Data constructor
    Tensor(const Eigen::MatrixXf& data, bool requires_grad = false);

    // ===== Autograd =====
    void backward();
    void zero_grad();

    // ===== Core Ops =====
    std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& other);
    std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& other);
    std::shared_ptr<Tensor> operator/(float scalar);
    std::shared_ptr<Tensor> operator*(float scalar);
    std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor>& other);

    std::shared_ptr<Tensor> relu(); // Consider adding GeLu
    std::shared_ptr<Tensor> sin();
    std::shared_ptr<Tensor> cos();

    // ===== Reshaping / Transpose =====
    std::shared_ptr<Tensor> transpose();
    std::shared_ptr<Tensor> reshape(int new_rows, int new_cols);

    // ===== Head Management =====
    std::vector<std::shared_ptr<Tensor>> split_heads(int num_heads);
    std::shared_ptr<Tensor> concat_heads(const std::vector<std::shared_ptr<Tensor>>& heads);


    // ===== Utilities =====
    std::pair<int, int> shape() const;
    void print(const std::string& label = "") const;
    std::shared_ptr<Tensor> clone() const;
    std::shared_ptr<Tensor> detach() const;
};
