#include "../incl/core/Tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>

void check_grad(const Eigen::MatrixXf& actual, const Eigen::MatrixXf& expected, const std::string& msg, float tolerance = 1e-6) {
    if (!actual.isApprox(expected, tolerance)) {
        std::cerr << "[FAIL] " << msg << "\nExpected:\n" << expected << "\nGot:\n" << actual << "\n";
        std::cerr << "Max difference: " << (actual - expected).cwiseAbs().maxCoeff() << std::endl;
        exit(1);
    } else {
        std::cout << "[PASS] " << msg << /* "\nExpected:\n" << expected << "\nGot:\n" << actual << */ "\n";
    }
}

void check_tensor(const std::shared_ptr<Tensor>& actual, const Eigen::MatrixXf& expected, const std::string& msg, float tolerance = 1e-6) {
    if (!actual->data.isApprox(expected, tolerance)) {
        std::cerr << "[FAIL] " << msg << "\nExpected:\n" << expected << "\nGot:\n" << actual->data << "\n";
        std::cerr << "Max difference: " << (actual->data - expected).cwiseAbs().maxCoeff() << std::endl;
        exit(1);
    } else {
        std::cout << "[PASS] " << msg << /*"\nExpected:\n" << expected << "\nGot:\n" << actual->data << */ "\n";
    }
}

int main() {
    std::cout << "=== Tensor Library Test Suite ===\n\n";

    // Test 1: Basic creation and shape
    std::cout << "Test 1: Basic creation and shape\n";
    auto t = std::make_shared<Tensor>(2, 3);
    assert(t->shape().first == 2 && t->shape().second == 3);
    check_tensor(t, Eigen::MatrixXf::Zero(2, 3), "Zero tensor creation");
    std::cout << "[PASS] Tensor shape check\n\n";

    // Test 2: Addition + backward
    std::cout << "Test 2: Addition + backward\n";
    auto a = std::make_shared<Tensor>(Eigen::MatrixXf::Ones(2, 2), true);
    auto b = std::make_shared<Tensor>(Eigen::MatrixXf::Constant(2, 2, 2.0), true);
    
    auto c = *a + b;
    check_tensor(c, Eigen::MatrixXf::Constant(2, 2, 3.0), "Addition forward");
    
    c->backward();
    check_grad(a->grad, Eigen::MatrixXf::Ones(2, 2), "Addition backward a");
    check_grad(b->grad, Eigen::MatrixXf::Ones(2, 2), "Addition backward b");
    std::cout << "\n";

    // Test 3: Scalar multiplication + backward
    std::cout << "Test 3: Scalar multiplication + backward\n";
    auto d = std::make_shared<Tensor>(Eigen::MatrixXf::Constant(2, 2, 3.0), true);
    auto e = *d * 2.0f;
    check_tensor(e, Eigen::MatrixXf::Constant(2, 2, 6.0), "Scalar multiplication forward");
    
    e->backward();
    check_grad(d->grad, Eigen::MatrixXf::Constant(2, 2, 2.0), "Scalar multiplication backward");
    std::cout << "\n";

    // Test 4: ReLU + backward
    std::cout << "Test 4: ReLU + backward\n";
    Eigen::MatrixXf relu_input(2, 2);
    relu_input << 1.0, -2.0, 3.0, -4.0;
    auto f = std::make_shared<Tensor>(relu_input, true);
    
    auto g = f->relu();
    Eigen::MatrixXf expected_relu(2, 2);
    expected_relu << 1.0, 0.0, 3.0, 0.0;
    check_tensor(g, expected_relu, "ReLU forward");
    
    g->backward();
    Eigen::MatrixXf relu_grad(2, 2);
    relu_grad << 1, 0, 1, 0;
    check_grad(f->grad, relu_grad, "ReLU backward");
    std::cout << "\n";

    // Test 5: Matrix multiplication + backward
    std::cout << "Test 5: Matrix multiplication + backward\n";
    Eigen::MatrixXf A(2, 3), B(3, 2);
    A << 1, 2, 3, 4, 5, 6;
    B << 7, 8, 9, 10, 11, 12;
    
    auto m1 = std::make_shared<Tensor>(A, true);
    auto m2 = std::make_shared<Tensor>(B, true);
    
    auto m3 = m1->matmul(m2);
    Eigen::MatrixXf expected_matmul(2, 2);
    expected_matmul << 58, 64, 139, 154;
    check_tensor(m3, expected_matmul, "Matrix multiplication forward");
    
    m3->backward();
    Eigen::MatrixXf dL_dC = Eigen::MatrixXf::Ones(2, 2);
    check_grad(m1->grad, dL_dC * B.transpose(), "Matrix multiplication backward a");
    check_grad(m2->grad, A.transpose() * dL_dC, "Matrix multiplication backward b");
    std::cout << "\n";

    // Test 6: Transpose + backward
    std::cout << "Test 6: Transpose + backward\n";
    Eigen::MatrixXf transpose_mat(2, 3);
    transpose_mat << 1, 2, 3, 4, 5, 6;
    
    auto original = std::make_shared<Tensor>(transpose_mat, true);
    auto transposed = original->transpose();
    
    Eigen::MatrixXf expected_forward = transpose_mat.transpose();
    check_tensor(transposed, expected_forward, "Transpose forward");
    
    transposed->backward();
    Eigen::MatrixXf expected_grad = Eigen::MatrixXf::Ones(2, 3);
    check_grad(original->grad, expected_grad, "Transpose backward");
    std::cout << "\n";

    // Test 7: Reshape
    std::cout << "Test 7: Reshape\n";
    Eigen::MatrixXf reshape_mat(2, 3);
    reshape_mat << 1, 2, 3, 4, 5, 6;
    // Ensure the tensor we reshape from is not a temporary
    auto to_reshape = std::make_shared<Tensor>(reshape_mat, false);
    auto reshaped = to_reshape->reshape(3, 2);
    
    // Eigen's reshape (via Map) is column-wise.
    // Data [1, 4, 2, 5, 3, 6] reshaped to 3x2 becomes:
    Eigen::MatrixXf expected_reshape(3, 2);
    expected_reshape << 1, 5,
                        4, 3,
                        2, 6;
    check_tensor(reshaped, expected_reshape, "Reshape operation");
    std::cout << "\n";

    // Test 8: Split and Concat Heads
    std::cout << "Test 8: Split and Concat Heads\n";
    Eigen::MatrixXf big_mat(2, 4);
    big_mat << 1, 2, 3, 4, 5, 6, 7, 8;
    auto big = std::make_shared<Tensor>(big_mat, false);
    
    auto heads = big->split_heads(2);
    assert(heads.size() == 2);
    
    Eigen::MatrixXf expected_head1(2, 2);
    expected_head1 << 1, 2, 5, 6;
    check_tensor(heads[0], expected_head1, "First head");
    
    Eigen::MatrixXf expected_head2(2, 2);
    expected_head2 << 3, 4, 7, 8;
    check_tensor(heads[1], expected_head2, "Second head");
    
    auto joined = big->concat_heads(heads);
    check_tensor(joined, big_mat, "Concat heads round-trip");
    std::cout << "\n";

    // Test 9: Complex chain with proper dimensions
    std::cout << "Test 9: Complex chain with proper dimensions\n";
    Eigen::MatrixXf x_data(2, 3);
    x_data << 1, -2, 3, -4, 5, -6;
    auto x = std::make_shared<Tensor>(x_data, true);
    
    Eigen::MatrixXf w_data(3, 2);
    w_data << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6;
    auto w = std::make_shared<Tensor>(w_data, true);
    
    auto y = x->relu()->matmul(w);
    auto z = y->relu();
    
    // Check forward pass
    Eigen::MatrixXf expected_relu_x(2, 3);
    expected_relu_x << 1, 0, 3, 0, 5, 0;
    check_tensor(x->relu(), expected_relu_x, "ReLU in chain");
    
    z->backward();
    
    // Check that gradients are computed (not zero)
    assert(x->grad.cwiseAbs().sum() > 0);
    assert(w->grad.cwiseAbs().sum() > 0);
    std::cout << "[PASS] Complex chain gradients computed\n\n";

    // Test 10: Subtraction + backward
    std::cout << "Test 10: Subtraction + backward\n";
    auto sub_a = std::make_shared<Tensor>(Eigen::MatrixXf::Constant(2, 2, 5.0), true);
    auto sub_b = std::make_shared<Tensor>(Eigen::MatrixXf::Constant(2, 2, 2.0), true);
    
    auto sub_c = *sub_a - sub_b;
    check_tensor(sub_c, Eigen::MatrixXf::Constant(2, 2, 3.0), "Subtraction forward");
    
    sub_c->backward();
    check_grad(sub_a->grad, Eigen::MatrixXf::Ones(2, 2), "Subtraction backward a");
    check_grad(sub_b->grad, Eigen::MatrixXf::Constant(2, 2, -1.0), "Subtraction backward b");
    std::cout << "\n";

    std::cout << "=== ALL TESTS PASSED ===\n";
    return 0;
}



/* Compilation:

g++ -std=c++17 -I third-party/Eigen -I ./incl src/core/Tensor.cpp tests/better_test.cpp -o test_tensor
./test_tensor

*/