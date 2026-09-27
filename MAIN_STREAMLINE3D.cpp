//////////////////////////////////////////////////////////////
// DATE: 2024 - 08 - 12
// Code written by Sébastien Leclaire(sebastien.leclaire@polymtl.ca)
// This code solves the 3D advection of massless particles along 
// the streamlines of a chosen analytical velocity field.
//////////////////////////////////////////////////////////////


#include "arrayfire.h"

#include "../LIB_IO/include_IO.h"
#include "../LIB_STACKOVERFLOW/include_stackoverflow.h"
#include "../LIB_TYPE/include_type.h"

#undef min
#undef max

typedef double T;

#include <iostream>
#include <iomanip> // std::setprecision
# define M_PI 3.14159265358979323846  /* pi */

const T EPS = std::numeric_limits<T>::epsilon();

af::array pos_analytical(const af::array& posInit, const T tPos, const T omega)
{
    return posInit * std::exp(-(std::exp(-tPos) * (std::sin(omega * tPos) + omega * cos(omega * tPos))) / (omega * omega + 1)) * std::exp(omega / (omega * omega + (T)1));
}

af::array pos_analytical_steadyState(const af::array& posInit, const T omega)
{
    return posInit * std::exp(omega / (omega * omega + (T)1));
}
af::array velocity(const af::array& pos, const T tPos, const T omega)
{

    return pos * (std::sin(omega * tPos) * std::exp(-tPos));
}

int main(int argc, char* argv[])
{
    //af::setBackend(AF_BACKEND_CPU);
    // af::setBackend(AF_BACKEND_OPENCL);
    af::setBackend(AF_BACKEND_CUDA);
    af::setDevice(0);
    af::info(); std::cout << std::endl;

    int nParticle = 100000; // This line can be changed only for the performance analysis, but should not be changed for the accuracy analysis (streamline3D.xlsx)

    /////////////// Do not change code between thoses lines below ///////////////

    T omegaX = (T)1 * M_PI / (T)3;
    T omegaY = (T)2 * M_PI / (T)3;
    T omegaZ = (T)3 * M_PI / (T)3;

    // Initial position of all particles
    af::setSeed(0);
    af::array xInit = af::randu(nParticle, 1, type::TYPE_AF<T>());
    af::array yInit = af::randu(nParticle, 1, type::TYPE_AF<T>());
    af::array zInit = af::randu(nParticle, 1, type::TYPE_AF<T>());

    /////////////// Do not change code between thoses lines above ///////////////

    // Select the constant time step.
    T dt = (T)0.01 / (T)1; //Diviser le dt par:{1, 2, 4, 8, 16}

    // Choix du schéma numérique:
    // 0 = Solution Analytique 
    // 1 = Euler Forward
    // 2 = Second Order Runge-Kutta (RK2)
    // 3 = Fourth Order Runge-Kutta (RK4)
    // 4 = Adams-Bashforth
    int solverType = 1; // <- Choix du schéma numérique à utiliser

    // Main time marching loop
    bool isSteadyStateReached = false;
    int iT = 0;
    T tPos = 0;
    af::array xPosLast = xInit;
    af::array yPosLast = yInit;
    af::array zPosLast = zInit;
    af::array xPosNew, yPosNew, zPosNew;

    // Variables stockant les vitesses du pas précédent pour Adams-Bashforth
    af::array uxLast, uyLast, uzLast;

    // Démarrage du chronomètre ArrayFire avant la boucle 
    af::timer startTimer = af::timer::start();

    while (isSteadyStateReached == false)
    {
        if (solverType == 0) {
            // Analytical solver for the position of the particles (at tPos + dt)
            xPosNew = pos_analytical(xInit, tPos + dt, omegaX);
            yPosNew = pos_analytical(yInit, tPos + dt, omegaY);
            zPosNew = pos_analytical(zInit, tPos + dt, omegaZ);
        }
        else if (solverType == 1) {
            // 1. Forward Euler scheme
            xPosNew = xPosLast + dt * velocity(xPosLast, tPos, omegaX);
            yPosNew = yPosLast + dt * velocity(yPosLast, tPos, omegaY);
            zPosNew = zPosLast + dt * velocity(zPosLast, tPos, omegaZ);
        }
        else if (solverType == 2) {
            // 2. Second Order Runge-Kutta scheme (RK2 aka midpoint)
            af::array k1x = dt * velocity(xPosLast, tPos, omegaX);
            af::array k1y = dt * velocity(yPosLast, tPos, omegaY);
            af::array k1z = dt * velocity(zPosLast, tPos, omegaZ);

            xPosNew = xPosLast + dt * velocity(xPosLast + k1x / (T)2, tPos + dt / (T)2, omegaX);
            yPosNew = yPosLast + dt * velocity(yPosLast + k1y / (T)2, tPos + dt / (T)2, omegaY);
            zPosNew = zPosLast + dt * velocity(zPosLast + k1z / (T)2, tPos + dt / (T)2, omegaZ);
        }
        else if (solverType == 3) {
            // 3. Fourth Order Runge-Kutta scheme (RK4)
            af::array k1x = dt * velocity(xPosLast, tPos, omegaX);
            af::array k1y = dt * velocity(yPosLast, tPos, omegaY);
            af::array k1z = dt * velocity(zPosLast, tPos, omegaZ);

            af::array k2x = dt * velocity(xPosLast + k1x / (T)2, tPos + dt / (T)2, omegaX);
            af::array k2y = dt * velocity(yPosLast + k1y / (T)2, tPos + dt / (T)2, omegaY);
            af::array k2z = dt * velocity(zPosLast + k1z / (T)2, tPos + dt / (T)2, omegaZ);

            af::array k3x = dt * velocity(xPosLast + k2x / (T)2, tPos + dt / (T)2, omegaX);
            af::array k3y = dt * velocity(yPosLast + k2y / (T)2, tPos + dt / (T)2, omegaY);
            af::array k3z = dt * velocity(zPosLast + k2z / (T)2, tPos + dt / (T)2, omegaZ);

            af::array k4x = dt * velocity(xPosLast + k3x, tPos + dt, omegaX);
            af::array k4y = dt * velocity(yPosLast + k3y, tPos + dt, omegaY);
            af::array k4z = dt * velocity(zPosLast + k3z, tPos + dt, omegaZ);

            xPosNew = xPosLast + (k1x + (T)2 * k2x + (T)2 * k3x + k4x) / (T)6;
            yPosNew = yPosLast + (k1y + (T)2 * k2y + (T)2 * k3y + k4y) / (T)6;
            zPosNew = zPosLast + (k1z + (T)2 * k2z + (T)2 * k3z + k4z) / (T)6;
        }
        else if (solverType == 4) {
            // 4. Two-step Adams-Bashforth scheme with an initial half time step 
            // and a forward Euler’s predictor.
            af::array ux = velocity(xPosLast, tPos, omegaX);
            af::array uy = velocity(yPosLast, tPos, omegaY);
            af::array uz = velocity(zPosLast, tPos, omegaZ);

            if (iT == 0) {
                // Initialisation avec demi-pas de temps et prédicteur d'Euler , puis Euler
                af::array x_half = xPosLast + (dt / (T)2) * ux;
                af::array y_half = yPosLast + (dt / (T)2) * uy;
                af::array z_half = zPosLast + (dt / (T)2) * uz;

                xPosNew = xPosLast + dt * velocity(x_half, tPos + dt / (T)2, omegaX);
                yPosNew = yPosLast + dt * velocity(y_half, tPos + dt / (T)2, omegaY);
                zPosNew = zPosLast + dt * velocity(z_half, tPos + dt / (T)2, omegaZ);
            }
            else {
                // Schéma de Adams-Bashforth 
                xPosNew = xPosLast + dt / (T)2 * ((T)3 * ux - uxLast);
                yPosNew = yPosLast + dt / (T)2 * ((T)3 * uy - uyLast);
                zPosNew = zPosLast + dt / (T)2 * ((T)3 * uz - uzLast);
            }
            // Mettre à jour les vitesses
            uxLast = ux;
            uyLast = uy;
            uzLast = uz;
        }

        // Next time step
        tPos += dt;
        iT += 1;

        // Check if steady state is reached every 50 time steps.
        if (iT % 50 == 0 && iT > 0)
        {
            // Calculate the difference L1 between the current position and the last position of all particles.
            T L1sum = af::sum<T>(af::abs(xPosNew - xPosLast) + af::abs(yPosNew - yPosLast) + af::abs(zPosNew - zPosLast));
            std::cout << std::setprecision(16) << "Step " << iT << ", L1sum: " << L1sum << std::endl;

            // Activate stopping criterion if steady state is reached
            if (L1sum < (T)100 * EPS)
                isSteadyStateReached = true;
        }

        // Eval ArrayFire JIT tree for the main variables because a recursion loop may cause OOM or performance issue.
        xPosNew.eval();
        yPosNew.eval();
        zPosNew.eval();

        // Also advance in time the last position of the particles
        xPosLast = xPosNew;
        yPosLast = yPosNew;
        zPosLast = zPosNew;
    }

    // Analytical solution at steady state (i.e. at t_\inf)
    af::array xFinal = pos_analytical_steadyState(xInit, omegaX);
    af::array yFinal = pos_analytical_steadyState(yInit, omegaY);
    af::array zFinal = pos_analytical_steadyState(zInit, omegaZ);

    // Error at steady state between computed and analytical positions
    T L2error = std::sqrt(af::sum<T>((xPosLast - xFinal) * (xPosLast - xFinal) + (yPosLast - yFinal) * (yPosLast - yFinal) + (zPosLast - zFinal) * (zPosLast - zFinal)));

    // Synchronisation 
    af::sync();
    // Arrêt du chronomètre
    double totalExecutionTime = af::timer::stop(startTimer);

    std::cout << "------------------------------------------\n";
    std::cout << "Solver Type : " << solverType << "\n";
    std::cout << "dt          : " << dt << "\n";
    std::cout << std::setprecision(16) << "L2error     : " << L2error << "\n";
    std::cout << std::setprecision(6) << "Time (s)    : " << totalExecutionTime << " s" << std::endl;

    return 0;
}
