#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../third-party/doctest.h"
#include "../incl/core/Tensor.h"
#include <memory>
#include <iostream>

TEST_CASE("Tensor basic creation and shape") {
    auto t = std::make_shared<Tensor>(2, 3);
    CHECK(t->shape().first == 2);
    CHECK(t->shape().second == 3);
}

TEST_CASE("Tensor addition and backward") {
    auto a = std::make_shared<Tensor>(Eigen::MatrixXf::Ones(2, 2), true);
    auto b = std::make_shared<Tensor>(Eigen::MatrixXf::Constant(2, 2, 2.0), true);

    auto c = *a + b;
    c->backward();

    auto expected = Eigen::MatrixXf::Ones(2, 2);

    std::cout << "a->grad:\\n" << a->grad << std::endl;
    std::cout << "b->grad:\\n" << b->grad << std::endl;
    std::cout << "expected:\\n" << expected << std::endl;

    CHECK(a->grad.isApprox(expected));
    CHECK(b->grad.isApprox(expected));
}

TEST_CASE("Tensor scalar multiplication and backward") {
    auto a = std::make_shared<Tensor>(Eigen::MatrixXf::Constant(2, 2, 3.0), true);
    auto c = *a * 2.0f;
    c->backward();

    auto expected = Eigen::MatrixXf::Constant(2, 2, 2.0);
    std::cout << "a->grad:\\n" << a->grad << std::endl;
    std::cout << "expected:\\n" << expected << std::endl;
    CHECK(a->grad.isApprox(expected));
}

TEST_CASE("Tensor relu and backward") {
    Eigen::MatrixXf values(2, 2);
    values << 1.0, -2.0,
              3.0, -4.0;

    auto a = std::make_shared<Tensor>(values, true);
    auto r = a->relu();
    r->backward();

    Eigen::MatrixXf expected(2, 2);
    expected << 1, 0,
                1, 0;

    std::cout << "a->grad:\\n" << a->grad << std::endl;
    std::cout << "expected:\\n" << expected << std::endl;
    CHECK(a->grad.isApprox(expected));
}

TEST_CASE("Tensor dot and backward") {
    Eigen::MatrixXf A(2, 3);
    A << 1, 2, 3,
         4, 5, 6;
    Eigen::MatrixXf B(3, 2);
    B << 7, 8,
         9, 10,
         11, 12;

    auto a = std::make_shared<Tensor>(A, true);
    auto b = std::make_shared<Tensor>(B, true);

    auto c = a->dot(b); // (2x3) · (3x2) = (2x2)
    c->backward();

    Eigen::MatrixXf dL_dC = Eigen::MatrixXf::Ones(2, 2);
    Eigen::MatrixXf dL_dA = dL_dC * B.transpose();
    Eigen::MatrixXf dL_dB = A.transpose() * dL_dC;

    std::cout << "a->grad:\\n" << a->grad << std::endl;
    std::cout << "Expected dL_dA:\\n" << dL_dA << std::endl;
    std::cout << "b->grad:\\n" << b->grad << std::endl;
    std::cout << "Expected dL_dB:\\n" << dL_dB << std::endl;

    CHECK(a->grad.isApprox(dL_dA));
    CHECK(b->grad.isApprox(dL_dB));
}


TEST_CASE("Tracing back a full conputation graph") {
    // To Do
}