#include "arrayfire.h"
#include "../LIB_IO/include_IO.h"
#include "../LIB_MEMORY/include_memory.h"
#include "../LIB_PERFORMANCE_ANALYZER/include_performanceAnalyzer.h"
#include "../LIB_STACKOVERFLOW/include_stackoverflow.h"
#include "../LIB_TYPE/include_type.h"
#include "../LIB_EIGEN/eigen-3.4.0/Eigen/Eigen"
#include "../LIB_EIGEN/eigen-3.4.0/Eigen/SVD"

#undef min
#undef max
#include <iostream>
#include <iomanip> // std::setprecision
#define M_PI 3.14159265358979323846  /* pi */
#include <vector>
#include <cmath>
#include <fstream> // Ajout pour l'exportation des fichiers CSV
#include <string>  // Ajout pour nommer les fichiers dynamiquement

typedef double T;
constexpr T EPS = std::numeric_limits<T>::epsilon();

// Fonction utilitaire pour le Gradient Conjugué (modifiée pour extraire l'historique)
template<typename T>
int SolveCG(af::array A, af::array b, af::array& x, std::vector<T>& res_history, int max_iter = 10000, double tol = 1e-10) {
    af::array r = b - af::matmul(A, x);
    af::array p = r;
    T rsold = af::sum(r * r).scalar<T>();

    // Enregistrement du résidu initial
    res_history.clear();
    res_history.push_back(std::sqrt(rsold));

    int iter = 0;
    for (; iter < max_iter; iter++) {
        af::array Ap = af::matmul(A, p);
        T pAp = af::sum(p * Ap).scalar<T>();
        T alpha = rsold / pAp;

        x = x + alpha * p;
        r = r - alpha * Ap;

        T rsnew = af::sum(r * r).scalar<T>();
        T current_res = std::sqrt(rsnew);

        // Enregistrement du résidu courant
        res_history.push_back(current_res);

        if (current_res < tol) break;

        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    }
    return iter;
}

template<typename T>
void NonPositive(int n) {
    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << "------------------ Non Positive Matrix ------------------" << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;

    // Construction de la matrice A (col-major pour ArrayFire)
    std::vector<T> h_A(n * n, 0.0);
    for (int i = 0; i < n; i++) {
        h_A[i + i * n] = 4.0; // Diagonale principale
        if (i > 0) h_A[i + (i - 1) * n] = -1.0;
        if (i < n - 1) h_A[i + (i + 1) * n] = -1.0;
    }
    // Modification pour rendre la matrice non symétrique définie positive
    h_A[0 + 0 * n] = -4.0;
    h_A[(n - 1) + 0 * n] = 4.0;

    af::array A(n, n, h_A.data());

    // Le vecteur b correspond à la somme de chaque ligne pour avoir x=1
    af::array b = af::sum(A, 1);

    // 1. TRANSFORMATION (Équations normales : A^T * A * x = A^T * b)
    af::array At = af::transpose(A);
    af::array A_sym = af::matmul(At, A);
    af::array b_sym = af::matmul(At, b);

    // Résolution avec le Gradient Conjugué sur le système transformé
    af::array x_cg = af::constant(0.0, n, 1, A.type());
    std::vector<T> res_cg_dummy; // Vecteur jetable pour la signature de fonction
    int iter_cg = SolveCG<T>(A_sym, b_sym, x_cg, res_cg_dummy);

    // 2. JACOBI sur le système original (risque de divergence)
    af::array D_orig = af::diag(A, 0, true);
    af::array D_mat_orig = af::diag(D_orig, 0, false);
    af::array R_orig = A - D_mat_orig;
    af::array x_jac_orig = af::constant(0.0, n, 1, A.type());

    int iter_jac_orig = 0;
    for (; iter_jac_orig < 1000; iter_jac_orig++) {
        x_jac_orig = (b - af::matmul(R_orig, x_jac_orig)) / D_orig;
        af::array r = b - af::matmul(A, x_jac_orig);
        if (std::sqrt(af::sum(r * r).scalar<T>()) < 1e-10) break;
    }

    // 3. JACOBI sur le système transformé (A^T A)
    af::array D_sym = af::diag(A_sym, 0, true);
    af::array D_mat_sym = af::diag(D_sym, 0, false);
    af::array R_sym = A_sym - D_mat_sym;
    af::array x_jac_sym = af::constant(0.0, n, 1, A.type());

    int iter_jac_sym = 0;
    for (; iter_jac_sym < 10000; iter_jac_sym++) {
        x_jac_sym = (b_sym - af::matmul(R_sym, x_jac_sym)) / D_sym;
        af::array r = b_sym - af::matmul(A_sym, x_jac_sym);
        if (std::sqrt(af::sum(r * r).scalar<T>()) < 1e-10) break;
    }

    std::cout << "Gradient Conjugue (Transforme A^T*A) - Iters : " << iter_cg << std::endl;
    std::cout << "Jacobi (Matrice Originale A)         - Iters : " << iter_jac_orig << " (Diverge si max iters atteint)" << std::endl;
    std::cout << "Jacobi (Transforme A^T*A)            - Iters : " << iter_jac_sym << std::endl;
}

template<typename T>
void IllConditionning(int n) {
    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << "--------------- Ill Conditionning Matrix-----------------" << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;

    std::vector<T> h_A(n * n, 0.0);
    for (int i = 0; i < n; i++) {
        T val = std::pow(10.0, -(double)n / 2.0 + (double)i);
        h_A[i + i * n] = val; // Diagonale principale
        if (i < n - 1) {
            h_A[i + (i + 1) * n] = val / 2.0; // Sur-diagonale
            h_A[(i + 1) + i * n] = val / 2.0; // Sous-diagonale
        }
    }

    af::array A(n, n, h_A.data());
    af::array b = af::sum(A, 1);

    // Calcul du Conditionnement avec Eigen (SVD)
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> Eigen_A(h_A.data(), n, n);
    Eigen::JacobiSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> svd(Eigen_A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    auto singularValues = svd.singularValues();
    T condNumber = singularValues(0) / singularValues(singularValues.size() - 1);

    std::cout << "Taille n = " << n << " | Conditionnement (Eigen SVD) : " << condNumber << std::endl;

    // 1. Gradient Conjugué Standard
    af::array x_cg = af::constant(0.0, n, 1, A.type());
    std::vector<T> res_cg; // Historique GC Standard
    int iter_cg = SolveCG<T>(A, b, x_cg, res_cg);

    // 2. Gradient Conjugué Préconditionné (PCG avec Jacobi M = diag(A))
    af::array D = af::diag(A, 0, true);
    af::array x_pcg = af::constant(0.0, n, 1, A.type());
    af::array r = b - af::matmul(A, x_pcg);
    af::array z = r / D; // Application du préconditionneur
    af::array p = z;

    std::vector<T> res_pcg; // Historique GC Préconditionné
    T r_norm_init = std::sqrt(af::sum(r * r).scalar<T>());
    res_pcg.push_back(r_norm_init);

    T rzold = af::sum(r * z).scalar<T>();
    int iter_pcg = 0;

    for (; iter_pcg < 10000; iter_pcg++) {
        af::array Ap = af::matmul(A, p);
        T pAp = af::sum(p * Ap).scalar<T>();
        T alpha = rzold / pAp;

        x_pcg = x_pcg + alpha * p;
        r = r - alpha * Ap;

        T r_norm = std::sqrt(af::sum(r * r).scalar<T>());
        res_pcg.push_back(r_norm); // Enregistrement du résidu courant

        if (r_norm < 1e-10) break;

        z = r / D; // Application du préconditionneur
        T rznew = af::sum(r * z).scalar<T>();

        p = z + (rznew / rzold) * p;
        rzold = rznew;
    }

    std::cout << "GC Standard        - Iters : " << iter_cg << std::endl;
    std::cout << "GC Preconditionne  - Iters : " << iter_pcg << std::endl;

    // Exportation des résultats vers un fichier CSV 
    std::string filename = "Residuals_n" + std::to_string(n) + ".csv";
    std::ofstream outfile(filename);
    if (outfile.is_open()) {
        outfile << "Iteration,GC_Standard,GC_Preconditionne\n";
        size_t max_iters = std::max(res_cg.size(), res_pcg.size());
        for (size_t i = 0; i < max_iters; i++) {
            outfile << i << ",";
            if (i < res_cg.size()) outfile << res_cg[i];
            outfile << ",";
            if (i < res_pcg.size()) outfile << res_pcg[i];
            outfile << "\n";
        }
        outfile.close();
        std::cout << ">> Historique des residus exporte dans " << filename << std::endl;
    }
    else {
        std::cerr << "Erreur: Impossible de creer le fichier " << filename << std::endl;
    }
}

int main(int argc, char* argv[])
{
    af::setBackend(AF_BACKEND_CPU);
    af::setDevice(0);
    af::info();
    std::cout << std::endl;

    int n = 20;

    NonPositive<T>(n); // Partie 2
    std::cout << std::endl;
    IllConditionning<T>(n); // Partie 3

    // Test de conditionnement pour différentes tailles
    std::cout << std::endl << "--- Evolution avec n ---" << std::endl;
    IllConditionning<T>(10);
    IllConditionning<T>(30);

    return 0;
}