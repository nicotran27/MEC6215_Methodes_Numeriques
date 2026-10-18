


#include "arrayfire.h"

#include "../LIB_IO/include_IO.h"
#include "../LIB_MEMORY/include_memory.h"
#include "../LIB_PERFORMANCE_ANALYZER/include_performanceAnalyzer.h"
#include "../LIB_STACKOVERFLOW/include_stackoverflow.h"
#include "../LIB_TYPE/include_type.h"

#undef min
#undef max

#include <iostream>
#include <iomanip> // std::setprecision
# define M_PI 3.14159265358979323846  /* pi */

typedef double T;

template <typename T>
af::array heat2D_analytical_naive_c(int nx, int ny, T u0, T L, T H, T time, T alpha)
{
	T* uAnalytical = new T[nx * ny];
	for (int iNode = 0; iNode < nx * ny; ++iNode)
		uAnalytical[iNode] = (T)0;

	for (T mm = 1; mm <= 100; mm++)
	{
		for (T nn = 1; nn <= 100; nn++)
		{
			T numerator = (T)4 * H * L * u0 * std::sin(M_PI * mm / (T)2) * std::sin(M_PI * mm / (T)2) * std::sin(M_PI * nn / (T)2) * std::sin(M_PI * nn / (T)2) / (mm * nn * M_PI * M_PI);
			T denominator = (H * L * (std::sin((T)2 * M_PI * mm) - (T)2 * M_PI * mm) * (std::sin((T)2 * M_PI * nn) - (T)2 * M_PI * nn)) / ((T)16 * mm * nn * M_PI * M_PI);
			T A_mmnn = numerator / denominator;
			T lambda_mmnn = (nn * M_PI / L) * (nn * M_PI / L) + (mm * M_PI / H) * (mm * M_PI / H);

			T expPart = std::exp(-alpha * lambda_mmnn * time);

			for (int iRow = 0; iRow < nx; iRow++)
				for (int iCol = 0; iCol < ny; iCol++)
				{
					int index = iRow * ny + iCol;

					T xPos = T(iRow) / (T(nx) - (T)1);
					T yPos = T(iCol) / (T(ny) - (T)1);

					T xPart = std::sin(nn * M_PI * xPos / L);
					T yPart = std::sin(mm * M_PI * yPos / H);

					uAnalytical[index] = uAnalytical[index] + A_mmnn * xPart * yPart * expPart;
				}
		}
	}

	af::array uAnalyticalArray = af::array(nx, ny, uAnalytical);
	delete[] uAnalytical;

	return uAnalyticalArray;
}

template <typename T>
af::array heat2D_numerical_naive_c(int nx, int ny, int nt, T dx, T dy, T dt, T alpha)
{
	// Allocation de la mémoire pour (u) et  (u_next)
	T* u = new T[nx * ny];
	T* u_next = new T[nx * ny];

	// Initialisation des conditions initiales et des conditions aux limites
	for (int iRow = 0; iRow < nx; iRow++) {
		for (int iCol = 0; iCol < ny; iCol++) {
			int index = iRow * ny + iCol;

			// Conditions aux limites : u = 0 sur les 4 bords du domaine
			if (iRow == 0 || iRow == nx - 1 || iCol == 0 || iCol == ny - 1) {
				u[index] = (T)0;
			}
			else {
				// Condition initiale : u = 1 strictement à l'intérieur
				u[index] = (T)1;
			}
			// Initialisation identique pour la grille du temps suivant
			u_next[index] = u[index];
		}
	}

	// Précalcul des constantes de diffusion r_x et r_y pour éviter 
	// de refaire les calculs à chaque itération dans la boucle
	T rx = alpha * dt / (dx * dx);
	T ry = alpha * dt / (dy * dy);

	// Boucle d'intégration temporelle (Schéma Euler explicite)
	for (int t = 0; t < nt; t++) {

		// Boucles spatiales sur les noeuds intérieurs 
		for (int iRow = 1; iRow < nx - 1; iRow++) {
			for (int iCol = 1; iCol < ny - 1; iCol++) {
				int index = iRow * ny + iCol;      // noeud central (i, j)
				int iN = (iRow - 1) * ny + iCol;   // noeud Nord    (i-1, j)
				int iS = (iRow + 1) * ny + iCol;   // noeud Sud     (i+1, j)
				int iW = iRow * ny + (iCol - 1);   // noeud Ouest   (i, j-1)
				int iE = iRow * ny + (iCol + 1);   // noeud Est     (i, j+1)

				// Résolution de l'équation de la chaleur
				// Avant d'ordre 1 en temps, et Centré d'ordre 2 en espace
				u_next[index] = u[index] + rx * (u[iN] - (T)2 * u[index] + u[iS])
					+ ry * (u[iW] - (T)2 * u[index] + u[iE]);
			}
		}

		// Mise à jour de la grille u pour le prochain pas de temps
		for (int i = 0; i < nx * ny; i++) {
			u[i] = u_next[i];
		}
	}

	// Conversion du tableau C/C++ natif vers un objet ArrayFire 
	af::array uSol = af::array(nx, ny, u);

	// Libération de la mémoire allouée 
	delete[] u;
	delete[] u_next;

	return uSol;
}



template <typename T>
af::array heat2D_numerical_arrayfire(int nx, int ny, int nt, T dx, T dy, T dt, T alpha)
{	// Initialisation de tout le domaine à u = 1 (condition initiale)
	af::array uSol = af::constant(1.0, nx, ny, type::TYPE_AF<T>());
	// Application des conditions aux limites à u = 0 sur les bords
	uSol(0, af::span) = 0;          // Bord haut 
	uSol(nx - 1, af::span) = 0;     // Bord bas 
	uSol(af::span, 0) = 0;          // Bord gauche 
	uSol(af::span, ny - 1) = 0;     // Bord droit 

	// Précalcul des coefficients
	T rx = alpha * dt / (dx * dx);
	T ry = alpha * dt / (dy * dy);
	// Création des séquences d'indices pour l'intérieur du domaine 
	// Ca équivaut aux boucles "for (int i = 1; i < nx - 1; i++)"
	af::seq I(1, nx - 2);
	af::seq J(1, ny - 2);

	af::seq I_minus(0, nx - 3);
	af::seq I_plus(2, nx - 1);
	af::seq J_minus(0, ny - 3);
	af::seq J_plus(2, ny - 1);

	// Boucle temporelle
	for (int t = 0; t < nt; t++) {
		// Mise à jour en utilisant les séquences 
		uSol(I, J) = uSol(I, J) +
			rx * (uSol(I_plus, J) - 2.0 * uSol(I, J) + uSol(I_minus, J)) +
			ry * (uSol(I, J_plus) - 2.0 * uSol(I, J) + uSol(I, J_minus));

		af::eval(uSol); 
	}

	return uSol;
}

template <typename T>
af::array heat2D_analytical_arrayfire(int nx, int ny, T u0, T L, T H, T time, T alpha)
{
	// Initialisation de la solution analytique
	af::array uSol = af::constant(0.0, nx, ny, type::TYPE_AF<T>());

	// Création de grilles contenant les numéros de lignes et colonnes
	// af::iota génère une séquence (0, 1, 2, ...). 
	af::array iRow = af::iota(af::dim4(nx, 1), af::dim4(1, ny), type::TYPE_AF<T>());
	af::array iCol = af::iota(af::dim4(1, ny), af::dim4(nx, 1), type::TYPE_AF<T>());

	// Conversion des indices en positions physiques (x et y) normalisées entre 0 et 1
	af::array xPos = iRow / (T(nx) - (T)1);
	af::array yPos = iCol / (T(ny) - (T)1);

	// Boucles sur les termes de la série de Fourier (M et N)
	for (T mm = 1; mm <= 100; mm++)
	{
		for (T nn = 1; nn <= 100; nn++)
		{
			// Calcul des coefficients A_mn constants pour la série de Fourier (identique à la version naïve)
			T numerator = (T)4 * H * L * u0 * std::sin(M_PI * mm / (T)2) * std::sin(M_PI * mm / (T)2) * std::sin(M_PI * nn / (T)2) * std::sin(M_PI * nn / (T)2) / (mm * nn * M_PI * M_PI);
			T denominator = (H * L * (std::sin((T)2 * M_PI * mm) - (T)2 * M_PI * mm) * (std::sin((T)2 * M_PI * nn) - (T)2 * M_PI * nn)) / ((T)16 * mm * nn * M_PI * M_PI);
			T A_mmnn = numerator / denominator;

			// Valeur propre lambda_mn
			T lambda_mmnn = (nn * M_PI / L) * (nn * M_PI / L) + (mm * M_PI / H) * (mm * M_PI / H);

			// Terme de décroissance exponentielle lié au temps
			T expPart = std::exp(-alpha * lambda_mmnn * time);

			// Calcul vectoriel du terme spatial (sinus) sur la matrice en même temps
			af::array xPart = af::sin(nn * M_PI * xPos / L);
			af::array yPart = af::sin(mm * M_PI * yPos / H);

			// Ajout du terme courant (m, n) à la solution globale
			uSol += A_mmnn * xPart * yPart * expPart;
		}
		af::eval(uSol);
	}

	return uSol;
}

template <typename T>
af::array heat2D_numerical_arrayfire_convolution(int nx, int ny, int nt, T dx, T dy, T dt, T alpha)
{
	// Initialisation
	af::array uSol = af::constant(1.0, nx, ny, type::TYPE_AF<T>());

	// Conditions aux limites
	uSol(0, af::span) = 0;
	uSol(nx - 1, af::span) = 0;
	uSol(af::span, 0) = 0;
	uSol(af::span, ny - 1) = 0;

	// Précalcul du coefficient
	T r = alpha * dt / (dx * dx); 
	T h_kernel[] = { 0,  r,  0,
					 r, -4 * r, r,
					 0,  r,  0 };
	af::array kernel = af::array(3, 3, h_kernel); 

	// Boucle temporelle
	for (int t = 0; t < nt; t++) {

		// Utilisation de kernel dans la convolution
		af::array diff = af::convolve2(uSol, kernel, AF_CONV_DEFAULT, AF_CONV_SPATIAL);
		uSol = uSol + diff;

		// Ré-application des conditions aux limites
		uSol(0, af::span) = 0;
		uSol(nx - 1, af::span) = 0;
		uSol(af::span, 0) = 0;
		uSol(af::span, ny - 1) = 0;

		af::eval(uSol);
	}

	return uSol;
}

int main(int argc, char *argv[])
{
	af::setBackend(AF_BACKEND_CUDA);
	af::setDevice(0);
	af::info(); std::cout << std::endl;

	// Physical domain size
	T L = 1;
	T H = L; 
	T tFinal = (T)0.01;

	// Number of grid point in x-direction, y-direction
	int nx = 801; // (51, 101, 201, 401, 801)
	int ny = 801;
	
	T dx = L / (T(nx) - (T)1); //  Spatial step in x
	T dy = dx; // Spatial step in y
	T alpha = (T)1; // Thermal diffusivity
	
	// Stability condition on the time step
	T r = (T)1 / (T)4;         
	T dt = r * dx * dx / alpha;  
	
	// Number of time steps
	int nt = std::ceil(tFinal / dt);
	
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Note that the content below does not require any modification.
	// Only the definition and content of the functions above the main program may require some modifications.
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Computation with analytical_naive_c implementation
	af::timer timer_analytical_naive_c = af::timer::start();
	af::array uSol_analytical_naive_c = heat2D_analytical_naive_c<T>(nx, ny, (T)1, L, H, tFinal, alpha);
	uSol_analytical_naive_c.eval();
	af::sync();
	printf("Elapsed seconds for computing uSol_analytical_naive_c: %g\n", af::timer::stop(timer_analytical_naive_c));
	IO::arrayToVTK<T>("uSol_analytical_naive_c", "uSol_analytical_naive_c", nx, ny, 1, uSol_analytical_naive_c);
	IO::arrayToFile<T>("uSol_analytical_naive_c", uSol_analytical_naive_c);

	// Computation with numerical_naive_c implementation
	af::timer timer_numerical_naive_c = af::timer::start();
	af::array uSol_numerical_naive_c = heat2D_numerical_naive_c<T>(nx, ny, nt, dx, dy, dt, alpha);
	uSol_numerical_naive_c.eval();
	af::sync();
	printf("Elapsed seconds for computing uSol_numerical_naive_c: %g\n", af::timer::stop(timer_numerical_naive_c));
	IO::arrayToVTK<T>("uSol_numerical_naive_c", "uSol_numerical_naive_c", nx, ny, 1, uSol_numerical_naive_c);
	IO::arrayToFile<T>("uSol_numerical_naive_c", uSol_numerical_naive_c);

	// Computation with numerical_arrayfire implementation
	af::timer timer_numerical_arrayfire = af::timer::start();
	af::array uSol_numerical_arrayfire = heat2D_numerical_arrayfire<T>(nx, ny, nt, dx, dy, dt, alpha);
	uSol_numerical_arrayfire.eval();
	af::sync();
	printf("Elapsed seconds for computing uSol_numerical_arrayfire: %g\n", af::timer::stop(timer_numerical_arrayfire));
	IO::arrayToVTK<T>("uSol_numerical_arrayfire", "uSol_numerical_arrayfire", nx, ny, 1, uSol_numerical_arrayfire);
	IO::arrayToFile<T>("uSol_numerical_arrayfire", uSol_numerical_arrayfire);

	// Computation with numerical_arrayfire_convolution implementation
	af::timer timer_numerical_arrayfire_convolution = af::timer::start();
	af::array uSol_numerical_arrayfire_convolution = heat2D_numerical_arrayfire_convolution(nx, ny, nt, dx, dy, dt, alpha);
	uSol_numerical_arrayfire_convolution.eval();
	af::sync();
	printf("Elapsed seconds for computing uSol_numerical_arrayfire_convolution: %g\n", af::timer::stop(timer_numerical_arrayfire_convolution));
	IO::arrayToVTK<T>("uSol_numerical_arrayfire_convolution", "uSol_numerical_arrayfire_convolution", nx, ny, 1, uSol_numerical_arrayfire_convolution);
	IO::arrayToFile<T>("uSol_numerical_arrayfire_convolution", uSol_numerical_arrayfire_convolution);

	// Computation with analytical_arrayfire implementation
	af::timer timer_analytical_arrayfire = af::timer::start();
	af::array uSol_analytical_arrayfire = heat2D_analytical_arrayfire<T>(nx, ny, (T)1, L, H, tFinal, alpha);
	uSol_analytical_arrayfire.eval();
	af::sync();
	printf("Elapsed seconds for computing uSol_analytical_arrayfire: %g\n", af::timer::stop(timer_analytical_arrayfire));
	IO::arrayToVTK<T>("uSol_analytical_arrayfire", "uSol_analytical_arrayfire", nx, ny, 1, uSol_analytical_arrayfire);
	IO::arrayToFile<T>("uSol_analytical_arrayfire", uSol_analytical_arrayfire);

	// L2 error computation between the analytical solution and the numerical solution for all three implementations
	T L2error_naive_c = std::sqrt(af::sum<T>((uSol_numerical_naive_c - uSol_analytical_naive_c) * (uSol_numerical_naive_c - uSol_analytical_naive_c)) * dx * dy);
	std::cout << std::setw(35) << "L2error_naive_c: " << std::setw(20) << std::setprecision(16) << L2error_naive_c << std::endl;
	T L2error_arrayfire = std::sqrt(af::sum<T>((uSol_numerical_arrayfire - uSol_analytical_naive_c) * (uSol_numerical_arrayfire - uSol_analytical_naive_c)) * dx * dy);
	std::cout << std::setw(35) << "L2error_arrayfire: " << std::setw(20) << std::setprecision(16) << L2error_arrayfire << std::endl;
	T L2error_arrayfire_convolution = std::sqrt(af::sum<T>((uSol_numerical_arrayfire_convolution - uSol_analytical_naive_c) * (uSol_numerical_arrayfire_convolution - uSol_analytical_naive_c)) * dx * dy);
	std::cout << std::setw(35) << "L2error_arrayfire_convolution: " << std::setw(20) << std::setprecision(16) << L2error_arrayfire_convolution << std::endl;

	// L1 norm difference between naive_c implementation and your implementation
	std::cout << std::setw(35) << "L1_diff_analytical: "            << std::setprecision(16) << af::sum<T>(af::abs(uSol_analytical_naive_c - uSol_analytical_arrayfire)) << std::endl;
	std::cout << std::setw(35) << "L1_diff_arrayfire: "             << std::setprecision(16) << af::sum<T>(af::abs(uSol_numerical_naive_c  - uSol_numerical_arrayfire)) << std::endl;
	std::cout << std::setw(35) << "L1_diff_arrayfire_convolution: " << std::setprecision(16) << af::sum<T>(af::abs(uSol_numerical_naive_c  - uSol_numerical_arrayfire_convolution)) << std::endl;

	//vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv//
	//vvvvvvv Write below the code you want for debugging or for any other reason vvvvvvvv//

	return 0;
}
