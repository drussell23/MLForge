#include "ml/algorithms/logistic_regression.h"
#include "ml/core/matrix.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <exception>
#include <stdexcept>

using namespace std;

// Helper function for floating-point comparisons.
bool approximatelyEqual(double a, double b, double tolerance = 1e-6);

// Function prototypes for test cases.
void testOrFunction();
void testAndFunction();
void testRegularization();
