#include "arrayfire.h"
#include "../LIB_IO/include_IO.h"
#include "../LIB_MEMORY/include_memory.h"
#include "../LIB_PERFORMANCE_ANALYZER/include_performanceAnalyzer.h"
#include "../LIB_STACKOVERFLOW/include_stackoverflow.h"
#include "../LIB_TYPE/include_type.h"
#include "../LIB_EIGEN\eigen-3.4.0\Eigen\Eigen"

#undef min
#undef max
#include <iostream>
#include <iomanip> // std::setprecision
#define M_PI 3.14159265358979323846  /* pi */
#include <vector>
#include <cmath>
#include <chrono>

typedef double T;
constexpr T EPS = std::numeric_limits<T>::epsilon();

template<typename T>
void Poisson2D(int nx) {
    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << "------------------------ Poisson ------------------------" << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;

    // Définition du domaine et des paramètres
    int ny = nx;
    T x_min = -1.0, x_max = 1.0;
    T y_min = -1.0, y_max = 1.0;
    T dx = (x_max - x_min) / (nx - 1);
    T dy = (y_max - y_min) / (ny - 1);

    // Noeuds intérieurs
    int n_in_x = nx - 2;
    int n_in_y = ny - 2;
    int N = n_in_x * n_in_y;

    // Allocation CPU
    std::vector<T> h_A(N * N, 0.0);
    std::vector<T> h_b(N, 0.0);
    std::vector<T> h_ua(N, 0.0);

    for (int j = 0; j < n_in_y; j++) {
        for (int i = 0; i < n_in_x; i++) {
            int row = j * n_in_x + i;

            // Remplissage de la matrice A 
            h_A[row * N + row] = 4.0;
            if (i > 0) h_A[(row - 1) * N + row] = -1.0;
            if (i < n_in_x - 1) h_A[(row + 1) * N + row] = -1.0;
            if (j > 0) h_A[(row - n_in_x) * N + row] = -1.0;
            if (j < n_in_y - 1) h_A[(row + n_in_x) * N + row] = -1.0;

            T x = x_min + (i + 1) * dx;
            T y = y_min + (j + 1) * dy;

            // Solution analytique
            h_ua[row] = std::exp(-x * x) + std::exp(-y * y);

            // Terme source 
            T f = std::exp(-x * x) * (4 * x * x - 2) + std::exp(-y * y) * (4 * y * y - 2);
            T b_val = -dx * dy * f;

            // Application des conditions limites de Dirichlet
            if (i == 0) b_val += std::exp(-x_min * x_min) + std::exp(-y * y);
            if (i == n_in_x - 1) b_val += std::exp(-x_max * x_max) + std::exp(-y * y);
            if (j == 0) b_val += std::exp(-x * x) + std::exp(-y_min * y_min);
            if (j == n_in_y - 1) b_val += std::exp(-x * x) + std::exp(-y_max * y_max);

            h_b[row] = b_val;
        }
    }

    try {
        
        af::array A(af::dim4(N, N), h_A.data());
        af::array b(af::dim4(N, 1), h_b.data());
        af::array ua(af::dim4(N, 1), h_ua.data());
        
        
        af::eval(A);
        af::eval(b);
        af::eval(ua);
        af::sync();


        // ---------------------------------------------------------
        // 1. Résolution avec la méthode directe
        // ---------------------------------------------------------
        auto start_direct = std::chrono::high_resolution_clock::now();

        af::array u_direct = af::solve(A, b);

        
        af::eval(u_direct);
        af::sync();

        auto end_direct = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_direct = end_direct - start_direct;

        T sum_err_direct = af::sum(af::pow(u_direct - ua, 2)).scalar<T>();
        T err_direct = std::sqrt(sum_err_direct * dx * dy);

        std::cout << "Solveur Direct :" << std::endl;
        std::cout << "  - Temps de calcul : " << duration_direct.count() << " ms" << std::endl;
        std::cout << "  - Erreur L2       : " << err_direct << std::endl;
        std::cout << "---------------------------------------------------------" << std::endl;

        // ---------------------------------------------------------
        // 2. Résolution avec la méthode itérative (Gradient Conjuguré)
        // ---------------------------------------------------------
        auto start_cg = std::chrono::high_resolution_clock::now();

        af::array x_cg = af::constant(0.0, af::dim4(N, 1), A.type());
        af::array r = b - af::matmul(A, x_cg);
        af::array p = r;

        T rsold = af::sum(r * r).scalar<T>();

        int max_iter = 10000;
        int iter = 0;

        for (int i = 0; i < max_iter; i++) {
            iter++;
            af::array Ap = af::matmul(A, p);

            T pAp = af::sum(p * Ap).scalar<T>();
            T alpha = rsold / pAp;

            x_cg = x_cg + alpha * p;
            r = r - alpha * Ap;

            T rsnew = af::sum(r * r).scalar<T>();
            if (std::sqrt(rsnew) < 1e-10) break; // Tolérance atteinte

            p = r + (rsnew / rsold) * p;
            rsold = rsnew;
        }

        
        af::eval(x_cg);
        af::sync();

        auto end_cg = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_cg = end_cg - start_cg;

        T sum_err_cg = af::sum(af::pow(x_cg - ua, 2)).scalar<T>();
        T err_cg = std::sqrt(sum_err_cg * dx * dy);

        std::cout << "Solveur Gradient Conjugue :" << std::endl;
        std::cout << "  - Temps de calcul : " << duration_cg.count() << " ms" << std::endl;
        std::cout << "  - Erreur L2       : " << err_cg << std::endl;
        std::cout << "  - Iterations      : " << iter << std::endl;
        std::cout << "---------------------------------------------------------" << std::endl;

    }
    catch (af::exception& e) {
        std::cerr << "Exception ArrayFire : " << e.what() << std::endl;
    }
}

int main(int argc, char* argv[])
{
    af::setBackend(AF_BACKEND_CUDA);
    af::setDevice(0);
    af::info();
    std::cout << std::endl;

    int nx = 80; // (5,10,20,40,80)
    Poisson2D<T>(nx);

    return 0;
}