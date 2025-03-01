#ifndef ML_SERIALIZATION_SERIALIZER_H
#define ML_SERIALIZATION_SERIALIZER_H

#include <fstream>
#include <string>
#include <vector>
#include "ml/core/matrix.h"

// ---------------------------------------------------------------------
// Macro to select the matrix type for serialization.
// By default, we use ml::core::Matrix2D. You can override this macro
// before including this header if needed.
// ---------------------------------------------------------------------
#ifndef SERIALIZER_MATRIX_TYPE
    #define SERIALIZER_MATRIX_TYPE ml::core::Matrix2D
#endif

// Alias for ease-of-use.
#define S_MATRIX_TYPE SERIALIZER_MATRIX_TYPE

namespace ml {
namespace serialization {

/**
 * @class Serializer
 * @brief Provides simple text-based serialization and deserialization methods.
 *
 * This class offers static methods to save and load matrices and vectors in a
 * plain text format. It can be extended in the future to support more advanced
 * formats (such as JSON or binary serialization) as MLForge evolves.
 */
class Serializer {
public:
    /**
     * @brief Serializes a matrix to a file.
     *
     * The file will contain the number of rows and columns on the first line,
     * followed by the matrix elements in row-major order.
     *
     * @param matrix The matrix to serialize.
     * @param filename The file path where the matrix should be saved.
     * @return True if the file was successfully written, false otherwise.
     */
    static bool saveMatrix(const S_MATRIX_TYPE& matrix, const std::string& filename) {
        std::ofstream ofs(filename);
        if (!ofs) {
            return false;
        }
        ofs << matrix.rows() << " " << matrix.cols() << "\n";
        for (std::size_t i = 0; i < matrix.rows(); ++i) {
            for (std::size_t j = 0; j < matrix.cols(); ++j) {
                ofs << matrix(i, j) << " ";
            }
            ofs << "\n";
        }
        return true;
    }
    
    /**
     * @brief Deserializes a matrix from a file.
     *
     * The file should contain the matrix dimensions on the first line, followed by
     * the matrix elements in row-major order.
     *
     * @param matrix A reference to a matrix object to populate.
     * @param filename The file path from which to load the matrix.
     * @return True if the file was successfully read and the matrix loaded, false otherwise.
     */
    static bool loadMatrix(S_MATRIX_TYPE& matrix, const std::string& filename) {
        std::ifstream ifs(filename);
        if (!ifs) {
            return false;
        }
        std::size_t rows, cols;
        ifs >> rows >> cols;
        matrix.resize(rows, cols);
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                ifs >> matrix(i, j);
            }
        }
        return true;
    }
    
    /**
     * @brief Serializes a vector of doubles to a file.
     *
     * The file will start with the vector size, followed by each element on a new line.
     *
     * @param vec The vector to serialize.
     * @param filename The file path where the vector should be saved.
     * @return True if the file was successfully written, false otherwise.
     */
    static bool saveVector(const std::vector<double>& vec, const std::string& filename) {
        std::ofstream ofs(filename);
        if (!ofs) {
            return false;
        }
        ofs << vec.size() << "\n";
        for (const auto& val : vec) {
            ofs << val << "\n";
        }
        return true;
    }
    
    /**
     * @brief Deserializes a vector of doubles from a file.
     *
     * The file should begin with the size of the vector, followed by each element on a new line.
     *
     * @param vec A reference to a vector to populate.
     * @param filename The file path from which to load the vector.
     * @return True if the file was successfully read and the vector loaded, false otherwise.
     */
    static bool loadVector(std::vector<double>& vec, const std::string& filename) {
        std::ifstream ifs(filename);
        if (!ifs) {
            return false;
        }
        std::size_t size;
        ifs >> size;
        vec.resize(size);
        for (std::size_t i = 0; i < size; ++i) {
            ifs >> vec[i];
        }
        return true;
    }
};

} // namespace serialization
} // namespace ml

#endif // ML_SERIALIZATION_SERIALIZER_H
