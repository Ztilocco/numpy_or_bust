#include <math.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <regex>
#include <string>
#include <tuple>
#include <typeinfo>
#include <vector>

using namespace std;

// Good diagram:
// https://engmrk.com/wp-content/uploads/2018/09/Image-Architecture-of-Convolutional-Neural-Network.png

bool be_random = true;

class Layer {
 public:
  virtual ~Layer() = default;

  vector<vector<vector<double>>> h(vector<vector<vector<double>>> x);

  // Helper functions
  static void rand_init(vector<vector<vector<double>>>& tensor, int height, int width) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        // use numbers between -10 and 10
        double n = (double)rand() / RAND_MAX;  // scales rand() to [0, 1].
        n = n * 2 - 1;
        tensor[i][j][0] = n;  // (possibly) change to use float to save memory
      }
    }
  }

  static void rand_init(vector<vector<double>>& matrix, int height, int width) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        // use numbers between -10 and 10
        double n = (double)rand() / RAND_MAX;  // scales rand() to [0, 1].
        n = n * 2 - 1;
        matrix[i][j] = n;  // (possibly) change to use float to save memory
      }
    }
  }

  static void rand_init(vector<double>& matrix, int length) {
    for (int i = 0; i < length; i++) {
      // use numbers between -10 and 10
      double n = (double)rand() / RAND_MAX;  // scales rand() to [0, 1].
      n = n * 2 - 1;
      matrix[i] = n;
    }
  }

  vector<double> static add_vectors(vector<double> a, vector<double> b) {
    vector<double> c(a.size(), 0);

    for (int i = 0; i < a.size(); i++) {
      c[i] = (a[i] + b[i]);
    }

    return c;
  }

  vector<double> static scalar_multiple(vector<double> a, double n) {
    vector<double> c(a.size(), 0);

    for (int i = 0; i < a.size(); i++) {
      c[i] = n * (a[i]);
    }

    return c;
  }

  vector<vector<double>> static add_matrices(vector<vector<double>> a, vector<vector<double>> b) {
    vector<vector<double>> c(a.size(), vector<double>(a[0].size(), 0));

    for (int i = 0; i < a.size(); i++) {
      for (int j = 0; j < a[0].size(); j++) {
        c[i][j] = (a[i][j] + b[i][j]);
      }
    }

    return c;
  }

  vector<vector<double>> static scalar_multiple(vector<vector<double>> a, double n) {
    vector<vector<double>> c(a.size(), vector<double>(a[0].size(), 0));

    for (int i = 0; i < a.size(); i++) {
      for (int j = 0; j < a[0].size(); j++) {
        c[i][j] = n * (a[i][j]);
      }
    }

    return c;
  }

  vector<vector<vector<double>>> static add_tensors(vector<vector<vector<double>>> a,
                                                    vector<vector<vector<double>>> b) {
    vector<vector<vector<double>>> c(a.size(), vector<vector<double>>(a[0].size(), vector<double>(a[0][0].size(), 0)));

    vector<vector<vector<double>>> c_column;
    for (int i = 0; i < a.size(); i++) {
      for (int j = 0; j < a[0].size(); j++) {
        for (int k = 0; k < a[0][0].size(); k++) {
          c[i][j][k] = (a[i][j][k] + b[i][j][k]);
        }
      }
    }

    return c;
  }

  vector<vector<vector<double>>> static scalar_multiple(vector<vector<vector<double>>> a, double n) {
    vector<vector<vector<double>>> c(a.size(), vector<vector<double>>(a[0].size(), vector<double>(a[0][0].size(), 0)));

    vector<vector<vector<double>>> c_column;
    for (int i = 0; i < a.size(); i++) {
      for (int j = 0; j < a[0].size(); j++) {
        for (int k = 0; k < a[0][0].size(); k++) {
          c[i][j][k] = n * (a[i][j][k]);
        }
      }
    }

    return c;
  }
};

class Conv : public Layer {
 public:
  int num_input_channels;
  int num_filters;
  vector<int> size_per_filter;
  vector<int> stride_per_filter;

  vector<vector<vector<double>>> filters;
  // TODO: Add a bias per filter.
  Conv(int num_input_channels, int num_filters, vector<int> size_per_filter, vector<int> stride_per_filter) {
    // TODO: Check if there is a better way to save these.
    this->num_input_channels = num_input_channels;
    this->num_filters = num_filters;
    this->size_per_filter = size_per_filter;
    this->stride_per_filter = stride_per_filter;

    for (int i = 0; i < num_filters; i++) {
      // Filters are square
      int height = size_per_filter[i];
      int width = size_per_filter[i];

      vector<vector<double>> filter(height, vector<double>(width, 0));
      Layer().rand_init(filter, height, width);
      this->filters.push_back(filter);
    }
  }

  // TODO: Write a test for this function if needed.
  vector<vector<vector<double>>> h(vector<vector<vector<double>>> a) {
    // Input and output is num_channels x height x width
    // First filter adds to the output of the first channel only, etc.

    // feature map (or activation map) is the output of one filter (or kernel or
    // detector)
    vector<vector<vector<double>>> output_block;
    for (int i = 0; i < num_filters; i++) {  // Should be embarrassingly parallel
      vector<vector<double>> feature_map = convolve(a, filters[i], stride_per_filter[i]);
      output_block.push_back(feature_map);
    }
    return output_block;
  }

  // static because this is a self-contained method
  vector<vector<double>> static convolve(vector<vector<vector<double>>> a, vector<vector<double>> filter, int stride) {
    // a is num_channels x height x width
    // Reference:
    // https://stats.stackexchange.com/questions/335321/in-a-convolutional-neural-network-cnn-when-convolving-the-image-is-the-opera

    int depth = a.size();
    int height = a[0].size();
    int width = a[0][0].size();

    int depth_of_a = a.size();
    vector<vector<double>> feature_map = _convolve(a[0], filter, stride);
    for (int i = 1; i < depth; i++) {
      vector<vector<double>> feature_map_for_depth = _convolve(a[i], filter, stride);
      feature_map = add_matrices(feature_map, feature_map_for_depth);
    }

    for (int depth = 0; depth < depth_of_a; depth++) {
    }

    return feature_map;
  }

  // Need to take into account stride.
  vector<vector<double>> static _convolve(vector<vector<double>> a, vector<vector<double>> filter, int stride) {
    // Height and width of the convolution.
    int c_width = (a.size() - filter.size()) / stride + 1;
    int c_height = (a[0].size() - filter.size()) / stride + 1;

    vector<vector<double>> convolved(c_width, (vector<double>(c_height, 0)));
    for (int i = 0; i < c_width; ++i) {
      for (int j = 0; j < c_height; ++j) {
        for (int x = 0; x < filter.size(); ++x) {
          for (int y = 0; y < filter[0].size(); ++y) {
            convolved[i][j] = convolved[i][j] + a[i * stride + x][j * stride + y] * filter[x][y];
          }
        }
      }
    }

    return convolved;
  }

  void static _convolve_test() {
    vector<vector<double>> a = {{1.0, 1.0, 1.0, 0.0, 0.0},
                                {0.0, 1.0, 1.0, 1.0, 0.0},
                                {0.0, 0.0, 1.0, 1.0, 1.0},
                                {0.0, 0.0, 1.0, 1.0, 0.0},
                                {0.0, 1.0, 1.0, 0.0, 0.0}};
    vector<vector<double>> filter = {{1.0, 0.0, 1.0}, {0.0, 1.0, 0.0}, {1.0, 0.0, 1.0}};

    vector<vector<double>> actual_output = _convolve(a, filter, 2);
    vector<vector<double>> expected_output = {{4.0, 4.0}, {2.0, 4.0}};

    for (int i = 0; i < actual_output.size(); i++) {
      for (int j = 0; j < actual_output[i].size(); j++) {
        cout << actual_output[i][j] << ",";
        if (actual_output[i][j] != expected_output[i][j]) {
          throw(string) "Test failed! " + (string) __FUNCTION__;
        }
      }
      cout << endl;
    }

    vector<vector<double>> a2 = {{1.0, 1.0, 1.0, 0.0, 0.0},
                                 {0.0, 1.0, 1.0, 1.0, 0.0},
                                 {0.0, 0.0, 1.0, 1.0, 1.0},
                                 {0.0, 0.0, 1.0, 1.0, 0.0},
                                 {0.0, 1.0, 1.0, 0.0, 0.0}};
    vector<vector<double>> filter2 = {{1.0}};

    vector<vector<double>> actual_output2 = _convolve(a2, filter2, 1);
    vector<vector<double>> expected_output2 = {{1.0, 1.0, 1.0, 0.0, 0.0},
                                               {0.0, 1.0, 1.0, 1.0, 0.0},
                                               {0.0, 0.0, 1.0, 1.0, 1.0},
                                               {0.0, 0.0, 1.0, 1.0, 0.0},
                                               {0.0, 1.0, 1.0, 0.0, 0.0}};

    for (int i = 0; i < actual_output2.size(); i++) {
      for (int j = 0; j < actual_output2[i].size(); j++) {
        cout << actual_output2[i][j] << ",";
        if (actual_output2[i][j] != expected_output2[i][j]) {
          throw(string) "Test failed! " + (string) __FUNCTION__;
        }
      }
      cout << endl;
    }
  }

  void static convolve_test() {
    vector<vector<vector<double>>> a = {{{1.0, 1.0, 1.0, 0.0, 0.0},
                                         {0.0, 1.0, 1.0, 1.0, 0.0},
                                         {0.0, 0.0, 1.0, 1.0, 1.0},
                                         {0.0, 0.0, 1.0, 1.0, 0.0},
                                         {0.0, 1.0, 1.0, 0.0, 0.0}},

                                        {{0.0, 1.0, 0.0, 1.0, 0.0},
                                         {0.0, 0.0, 1.0, 1.0, 1.0},
                                         {0.0, 0.0, 1.0, 1.0, 0.0},
                                         {0.0, 1.0, 1.0, 0.0, 1.0},
                                         {0.0, 1.0, 1.0, 0.0, 0.0}},

                                        {{1.0, 0.0, 0.0, 0.0, 0.0},
                                         {0.0, 1.0, 0.0, 0.0, 0.0},
                                         {0.0, 0.0, 1.0, 0.0, 0.0},
                                         {0.0, 0.0, 0.0, 1.0, 0.0},
                                         {0.0, 0.0, 0.0, 0.0, 1.0}}};
    vector<vector<double>> filter = {{1, 0}, {1, 1}};

    vector<vector<double>> actual_output = convolve(a, filter, 2);
    vector<vector<double>> expected_output = {{4.0, 5.0}, {1.0, 7.0}};

    for (int i = 0; i < actual_output.size(); i++) {
      for (int j = 0; j < actual_output[i].size(); j++) {
        cout << actual_output[i][j] << ",";
        if (actual_output[i][j] != expected_output[i][j]) {
          throw(string) "Test failed! " + (string) __FUNCTION__;
        }
      }
      cout << endl;
    }

    vector<vector<vector<double>>> a2 = {{{9.0, 1.0, 9.0, 0.0, 9.0},
                                          {0.0, 1.0, 9.0, 1.0, 0.0},
                                          {9.0, 0.0, 9.0, 1.0, 9.0},
                                          {0.0, 0.0, 9.0, 0.0, 0.0},
                                          {9.0, 1.0, 9.0, 0.0, 9.0}}};
    vector<vector<double>> filter2 = {{1.0}};

    vector<vector<double>> actual_output2 = convolve(a2, filter2, 2);
    vector<vector<double>> expected_output2 = {{9.0, 9.0, 9.0}, {9.0, 9.0, 9.0}, {9.0, 9.0, 9.0}};

    for (int i = 0; i < actual_output2.size(); i++) {
      for (int j = 0; j < actual_output2[0].size(); j++) {
        cout << actual_output2[i][j] << ",";
        if (actual_output2[i][j] != expected_output2[i][j]) {
          throw(string) "Test failed! " + (string) __FUNCTION__;
        }
      }
      cout << endl;
    }
  }
};

class Pool : public Layer {};

class MaxPool : public Pool {
 public:
  int height;
  int width;
  int stride;

  // No num_input_channels variable is necessary because no weights are
  // allocated for Pooling
  MaxPool(int size) {
    this->height = size;
    this->width = size;
    this->stride = size;
  }

  vector<vector<vector<double>>> h(vector<vector<vector<double>>> a) {
    vector<vector<vector<double>>> output_block;

    int num_input_channels = a.size();

    for (int i = 0; i < num_input_channels; i++) {                               // Should be embarrassingly parallel
      vector<vector<double>> pool_map = _max_pool(a[i], height, width, stride);  // Max pool by later
      output_block.push_back(pool_map);
    }
    return output_block;
  }

  vector<vector<double>> static _max_pool(vector<vector<double>> a, int height, int width, int stride) {
    int i = 0;
    int j = 0;

    vector<vector<double>> pool_map;
    while (i <= a.size() - height) {
      vector<double> pooled_row;

      while (j <= a[0].size() - width) {
        double max_value = numeric_limits<double>::lowest();
        for (int x = 0; x < height && i + x < a.size(); ++x) {
          for (int y = 0; y < width && j + y < a[0].size(); ++y) {
            if (a[i + x][j + y] > max_value) {
              max_value = a[i + x][j + y];
            }
          }
        }
        pooled_row.push_back(max_value);
        j = j + stride;
      }
      pool_map.push_back(pooled_row);
      j = 0;
      i = i + stride;
    }
    return pool_map;
  }

  void static _max_pool_test() {
    vector<vector<double>> a = {{0, 1, 2, 3}, {4, 5, 6, 7}, {1, 1, 1, 1}, {9, 0, 6, 3}};

    vector<vector<double>> test_val = _max_pool(a, 2, 2, 2);
    vector<vector<double>> expected_val = {{5, 7}, {9, 6}};
    for (int i = 0; i < test_val.size(); ++i) {
      for (int j = 0; j < test_val[0].size(); ++j) {
        cout << test_val[i][j] << ",";
        if (test_val[i][j] != expected_val[i][j]) {
          throw(string) "Test failed! " + (string) __FUNCTION__;
        }
      }
      cout << endl;
    }

    vector<vector<double>> a2 = {{0, 1, 2, 3}, {4, 5, 6, 7}, {1, 1, 1, 1}, {9, 0, 6, 3}};

    vector<vector<double>> test_val2 = _max_pool(a2, 1, 2, 3);
    vector<vector<double>> expected_val2 = {{1}, {9}};
    for (int i = 0; i < test_val2.size(); ++i) {
      for (int j = 0; j < test_val2[0].size(); ++j) {
        cout << test_val2[i][j] << ",";
        if (test_val2[i][j] != expected_val2[i][j]) {
          throw(string) "Test failed! " + (string) __FUNCTION__;
        }
      }
      cout << endl;
    }
  }
};

class Act : public Layer {
 public:
  vector<vector<vector<double>>> h(vector<vector<vector<double>>> z) {
    // Applied the sigmoid element wise.
    vector<vector<vector<double>>> output_block;
    for (int i = 0; i < z.size(); i++) {
      vector<vector<double>> depth_output;
      for (int j = 0; j < z[0].size(); j++) {
        vector<double> row_output;
        for (int k = 0; k < z[0][0].size(); k++) {
          double activation = activation_func(z[i][j][k]);
          row_output.push_back(activation);
        }
        depth_output.push_back(row_output);
      }
      output_block.push_back(depth_output);
    }
    return output_block;
  }

  vector<vector<vector<double>>> da_dz(vector<vector<vector<double>>> z) {
    // Applied the sigmoid element wise.
    vector<vector<vector<double>>> output_block_partials;
    for (int i = 0; i < z.size(); i++) {
      vector<vector<double>> depth_output;
      for (int j = 0; j < z[0].size(); j++) {
        vector<double> row_output;
        for (int k = 0; k < z[0][0].size(); k++) {
          double activation = activation_func_derivative(z[i][j][k]);
          row_output.push_back(activation);
        }
        depth_output.push_back(row_output);
      }
      output_block_partials.push_back(depth_output);
    }
    return output_block_partials;
  }

  virtual double activation_func(double z) = 0;
  virtual double activation_func_derivative(double z) = 0;
};

class Sigmoid : public Act {
 public:
  double activation_func(double z) { return 1 / (1 + exp(-z)); }

  double activation_func_derivative(double z) { return activation_func(z) * (1 - activation_func(z)); };

  void static sigmoid_test() {
    vector<vector<vector<double>>> z = {{{1, 1}, {2, 2}, {3, 3}}, {{1, 0}, {0, 1}, {1, -1}}};
    vector<vector<vector<double>>> val = Sigmoid().h(z);
    vector<vector<vector<double>>> expected = {{{0.731059, 0.731059}, {0.880797, 0.880797}, {0.952574, 0.952574}},
                                               {{0.731059, 0.5}, {0.5, 0.731059}, {0.731059, 0.268941}}};

    for (int i = 0; i < z.size(); i++) {
      for (int j = 0; j < z[0].size(); j++) {
        for (int k = 0; k < z[0][0].size(); k++) {
          cout << val[i][j][k] << ", ";
          if (abs(val[i][j][k] - expected[i][j][k]) > 0.0001) {
            throw(string) "Test failed! " + (string) __FUNCTION__;
          }
        }
      }
    }

    cout << endl;
  }
};

class Relu : public Act {
 public:
  double activation_func(double z) { return max(0.0, z); }

  double activation_func_derivative(double z) {
    if (z > 0) {
      return 1;
    } else {
      return 0;
    }
  };

  void static relu_test() {
    vector<vector<vector<double>>> z = {{{1, 1}, {2, -2}, {3, -3}}, {{1, 0}, {0, 1}, {1, -1}}};
    vector<vector<vector<double>>> val = Relu().h(z);
    vector<vector<vector<double>>> expected = {{{1, 1}, {2, 0}, {3, 0}}, {{1, 0}, {0, 1}, {1, 0}}};

    for (int i = 0; i < z.size(); i++) {
      for (int j = 0; j < z[0].size(); j++) {
        for (int k = 0; k < z[0][0].size(); k++) {
          cout << val[i][j][k] << ", ";
          if (val[i][j][k] != expected[i][j][k]) {
            throw(string) "Test failed! " + (string) __FUNCTION__;
          }
        }
      }
    }

    cout << endl;
  }
};

class Flatten : public Layer {
  // Flattens to a column vector
 public:
  vector<vector<vector<double>>> static f(vector<vector<vector<double>>> a) {
    vector<vector<vector<double>>> flattened;
    for (int i = 0; i < a.size(); i++) {
      for (int j = 0; j < a[0].size(); j++) {
        for (int k = 0; k < a[0][0].size(); k++) {
          vector<vector<double>> num = {{a[i][j][k]}};
          flattened.push_back(num);  // Add a one element row vector to the column
        }
      }
    }
    return flattened;
  }
};

class Dense : public Layer {
 public:
  int num_out;
  int num_in;

  vector<vector<vector<double>>> weights;
  vector<double> biases;

  Dense(int num_out, int num_in) {
    this->num_out = num_out;
    this->num_in = num_in;

    // Initialize weights with all values zero, then set all weights to a random value
    this->weights = vector<vector<vector<double>>>(num_out, vector<vector<double>>(num_in, vector<double>(1, 0)));
    rand_init(weights, num_out, num_in);

    // Initialize biases with all values zero, then set all biases to a random value
    this->biases = vector<double>(num_out, 0);
    rand_init(biases, num_out);
  }

  vector<vector<vector<double>>> h(vector<vector<vector<double>>> a) {
    vector<vector<vector<double>>> zs;

    if (a.size() != num_in) {
      throw(string) "Mismatch between Dense parameters and incoming vector!";
    }

    for (int i = 0; i < num_out; i++) {
      double z = biases[i];
      for (int j = 0; j < num_in; j++) {
        z = z + weights[i][j][0] * a[j][0][0];
      }
      vector<vector<double>> num = {{z}};
      zs.push_back(num);
    }

    return zs;
  }

  void static h_test() {
    vector<vector<vector<double>>> a{{{1}}, {{2}}, {{3}}};  // e.g. a[0] = {{1}};

    Dense d = Dense(5, 3);
    d.weights = {{{1}, {0}, {0}}, {{0}, {1}, {0}}, {{0}, {0}, {1}}, {{0}, {0}, {0}}, {{0}, {0}, {0}}};
    d.biases = {0, 0, 0, 0, 0};

    vector<vector<vector<double>>> output = d.h(a);
    vector<vector<vector<double>>> expected_output = {{{1}}, {{2}}, {{3}}, {{0}}, {{0}}};
    for (int i = 0; i < output.size(); i++) {
      if (output[i][0][0] != expected_output[i][0][0]) {
        throw(string) "Test failed! " + (string) __FUNCTION__;
      }
    }

    vector<vector<vector<double>>> a2{{{1}}, {{2}}, {{3}}};

    Dense d2 = Dense(5, 3);
    d2.weights = {{{1}, {1}, {0}}, {{0}, {1}, {3}}, {{0}, {0}, {1}}, {{1}, {0}, {0}}, {{0}, {2}, {0}}};
    d2.biases = {0, 0, 0, 0, 0};

    vector<vector<vector<double>>> output2 = d2.h(a2);
    vector<vector<vector<double>>> expected_output2 = {{{3}}, {{11}}, {{3}}, {{1}}, {{4}}};
    for (int i = 0; i < output2.size(); i++) {
      if (output2[i][0][0] != expected_output2[i][0][0]) {
        throw(string) "Test failed! " + (string) __FUNCTION__;
      }
    }

    vector<vector<vector<double>>> a3{{{1}}, {{2}}, {{3}}};

    Dense d3 = Dense(5, 3);
    d3.weights = {{{1}, {1}, {0}}, {{0}, {1}, {3}}, {{0}, {0}, {1}}, {{1}, {0}, {0}}, {{0}, {2}, {0}}};
    d3.biases = {1, 1, 1, 2, -1};

    vector<vector<vector<double>>> output3 = d3.h(a3);
    vector<vector<vector<double>>> expected_output3 = {{{4}}, {{12}}, {{4}}, {{3}}, {{3}}};
    for (int i = 0; i < output2.size(); i++) {
      cout << output3[i][0][0] << endl;
      if (output3[i][0][0] != expected_output3[i][0][0]) {
        throw(string) "Test failed! " + (string) __FUNCTION__;
      }
    }
  }
};

class ConvNet {
 public:
  vector<Layer*> layers;
  vector<vector<vector<vector<double>>>> a;
  map<int, int> layer_map;

  ConvNet(vector<Layer*> layers) { this->layers = layers; }

  vector<vector<vector<double>>> h(vector<vector<vector<double>>> x) {
    a.clear();  // Start with an empty vector of activations

    vector<vector<vector<double>>> feature_map = x;
    // as.push_back(a);

    int l = 0;
    for (int L = 0; L < layers.size(); L++) {
      Layer* layer = layers[L];

      vector<vector<vector<double>>> z = feature_map;
      if (Conv* conv = dynamic_cast<Conv*>(layer)) {
        feature_map = conv->h(z);
        layer_map[l] = L;
        l++;
      } else if (MaxPool* pool = dynamic_cast<MaxPool*>(layer)) {
        feature_map = pool->h(z);
      } else if (Act* act = dynamic_cast<Act*>(layer)) {
        feature_map = act->h(z);
      } else if (Flatten* flatten = dynamic_cast<Flatten*>(layer)) {
        feature_map = flatten->f(z);
      } else if (Dense* dense = dynamic_cast<Dense*>(layer)) {
        feature_map = dense->h(z);
        layer_map[l] = L;
        l++;
      }

      a.push_back(feature_map);
    }

    return feature_map;
  }

  int predict(vector<vector<vector<double>>> x) {
    vector<vector<vector<double>>> feature_map = h(x);

    // Take argmax of the output
    int label = 0;
    // cout << feature_map[0][0][0] << ",";
    for (int i = 1; i < feature_map.size(); i++) {
      // cout << feature_map[i][0][0] << ",";
      if (feature_map[label][0][0] < feature_map[i][0][0]) {
        label = i;
      }
    }
    // cout << endl;

    return label;
  }

  void fit(vector<vector<vector<vector<double>>>> X, int Y[]) {
    /* Fit function.

    This is the gradient descent function.

    (1) Take a minibatch of examples (20%)
    (2) Run each example through _calc_dLoss_dParam and average out the gradient for the minibatch examples
    (3) Take a step alpha from current weights and biases towards the directions of the gradient
    (4) Repeat steps 2-4 until some convergence criteria
    (5) Evaluate the Loss every so often

    */

    int num_steps = 100;
    double alpha = 0.01;
    double minibatch_ratio = 0.1;

    for (int i = 0; i < num_steps; i++) {
      vector<int> batch = take_minibatch(X.size(), minibatch_ratio);

      if (i % 10 == 0) {
        cout << "Step: " << i << ". Loss is " << TotalLoss(X, Y) << ". Accuracy is " << TotalAccuracy(X, Y) << endl;
      }

      vector<tuple<vector<vector<vector<double>>>, vector<double>>> dParam_acc;

      for (int j = 0; j < batch.size(); j++) {
        h(X[batch[j]]);  // Saves a bunch of variables that we need for the backward pass
        vector<tuple<vector<vector<vector<double>>>, vector<double>>> dParam_per_layer =
            _calc_dLoss_dParam(Y[batch[j]]);

        if (j == 0) {
          dParam_acc = dParam_per_layer;
        } else {
          for (int k = 0; k < dParam_per_layer.size(); k++) {
            tuple<vector<vector<vector<double>>>, vector<double>> dParam = dParam_per_layer[k];

            // Do the accumulation for weights
            vector<vector<vector<double>>> weights = get<0>(dParam);
            get<0>(dParam_acc[k]) = Layer::add_tensors(get<0>(dParam_acc[k]), weights);

            // Do the accumulation for biases
            vector<double> biases = get<1>(dParam);
            get<1>(dParam_acc[k]) = Layer::add_vectors(get<1>(dParam_acc[k]), biases);
          }
        }
      }
      for (int k = 0; k < dParam_acc.size(); k++) {
        get<0>(dParam_acc[k]) = Layer::scalar_multiple(get<0>(dParam_acc[k]), batch.size());
        get<1>(dParam_acc[k]) = Layer::scalar_multiple(get<1>(dParam_acc[k]), batch.size());
      }

      // Add tensor to weights (first part of the tuple) and vector (second part of tuple) to biases
      // Do this for each layer (that's why dParam is a vector)

      // Add dParam to dParam_acc;

      int k = 0;
      for (int L = layers.size() - 1; L >= 0; L--) {
        bool is_last_output_box = false;
        Layer* layer = layers[L];
        if (Conv* conv = dynamic_cast<Conv*>(layer)) {
        } else if (MaxPool* pool = dynamic_cast<MaxPool*>(layer)) {
        } else if (Act* act = dynamic_cast<Act*>(layer)) {
        } else if (Flatten* flatten = dynamic_cast<Flatten*>(layer)) {
        } else if (Dense* dense = dynamic_cast<Dense*>(layer)) {
          dense->weights =
              Layer::add_tensors(dense->weights, Layer::scalar_multiple(get<0>(dParam_acc[k]), -1 * alpha));

          dense->biases = Layer::add_vectors(dense->biases, Layer::scalar_multiple(get<1>(dParam_acc[k]), -1 * alpha));

          k += 1;
        }
      }
    }
    cout << "Step: " << num_steps-1 << ". Loss is " << TotalLoss(X, Y) << ". Accuracy is " << TotalAccuracy(X, Y) << endl;
  }

  vector<int> take_minibatch(int N, double r) {
    vector<int> v(N);                     // vector with 100 ints.
    iota(std::begin(v), std::end(v), 0);  // Fill with 0, 1, ..., 99.

    int k = floor(N * r);  // size of each batch

    vector<int> out;
    size_t nelems = k;
    if (be_random) {
      std::sample(v.begin(), v.end(), std::back_inserter(out), nelems, std::mt19937{std::random_device{}()});
    } else {
      std::sample(v.begin(), v.end(), std::back_inserter(out), nelems, std::mt19937{});
    }

    return out;
  }

  // Calculate the accuracy per example
  double Accuracy(vector<vector<vector<double>>> x, int y) {
    if (y == 10) {
      throw(string) "Mismatch between label definition in Loss and incoming label!";
    }
    int label = predict(x);

    if (label == y) {
      return 1;
    } else {
      return 0;
    }
  }

  // Calculate the accuracy
  double TotalAccuracy(vector<vector<vector<vector<double>>>> X, int Y[]) {
    double acc{0};

    for (int i = 0; i < X.size(); i++) {
      acc += Accuracy(X[i], Y[i]);
    }
    return acc;
  }

  // Calculate the loss function
  double Loss(vector<vector<vector<double>>> x, int y) {
    if (y == 10) {
      throw(string) "Mismatch between label definition in Loss and incoming label!";
    }
    vector<double> y_vector(10, 0);
    y_vector[y] = 1;

    vector<vector<vector<double>>> feature_map = h(x);
    double acc{0};

    for (int i = 0; i < feature_map.size(); i++) {
      acc += (feature_map[i][0][0] - y_vector[i]) * (feature_map[i][0][0] - y_vector[i]) / 2;
    }
    return acc;
  }

  // Calculate the loss function
  double TotalLoss(vector<vector<vector<vector<double>>>> X, int Y[]) {
    double acc{0};

    for (int i = 0; i < X.size(); i++) {
      acc += Loss(X[i], Y[i]);
    }
    return acc;
  }

  vector<tuple<vector<vector<vector<double>>>, vector<double>>> _calc_dLoss_dParam(int y) {
    /*
    Return data type:
    Vector of a tuple of gradients, one per layer
    First thing in the tuple is the tensor of weight derivatives
    Second thing in the tuple is the vector of bias derivatives.

    NOTE: vector of biases might become a matrix of biases for the convolution layer.
    */

    vector<vector<vector<vector<double>>>> da_L_dz_L_per_layer(layers.size(), {{{}}});

    vector<double> y_vector(10, 0);
    y_vector[y] = 1;

    vector<tuple<vector<vector<vector<double>>>, vector<double>>> dParam_per_layer;

    map<vector<int>, double> layer_neuron_to_sensitivity;

    for (int L = layers.size() - 1; L >= 0; L--) {
      bool is_last_output_box = false;
      Layer* layer = layers[L];
      if (Conv* conv = dynamic_cast<Conv*>(layer)) {
        // a = conv->h(a);
      } else if (MaxPool* pool = dynamic_cast<MaxPool*>(layer)) {
        // a = pool->h(a);
      } else if (Act* act = dynamic_cast<Act*>(layer)) {
        /*
        For sigmoid: g'(a)
        */
        da_L_dz_L_per_layer[L - 1] = act->da_dz(a[L - 1]);
      } else if (Flatten* flatten = dynamic_cast<Flatten*>(layer)) {
        // v = flatten->f(a);
        // is_last_output_box = true;
      } else if (Dense* dense = dynamic_cast<Dense*>(layer)) {
        /*
        Dense + Act:
        dLoss/dW_(output_sigmal, incoming_signal)

        dLoss/dW_{i,j}^L = dz_i^L/dW_{i,j}^L * [da_i^L/dz_i^L * dLoss/da_i^L]

        dLoss/dW_{i,j}^{L-1} = dz_i^{L-1}/dW_{i,j}^{L-1} * da_i^{L-1}/dz_i^{L-1} * ...
          \sum_k^{num_neurons_in_L} (dz_k^L/da_i^{L-1} * [da_k^L/dz_k^L * ...])

        L(x, y) = \sum_i^I 1/2(a_i^L - y_i)^2
        a^L = g(z^L)
        z^L = W^L*a^{L-1}

        dz_i^L/dw_{i,j}^L = a_j^{L-1}           ok
        dz_i^{L-1}/dW_{i,j}^{L-1} = a_j^{L-2}   ok

        da_i^L/dz_i^L = (above)                 ok
        da_i^{L-1}/dz_i^{L-1} = (above)         ok

        dz_k^L/da_i^{L-1} = W_{k,i}^L           ok

        dLoss/da_i^L = (a_i^L - y_i)            ok

        dLoss/dB_i^L = dz_i^L/dB_i^L * [da_i^L/dz_i^L * dLoss/da_i^L]
        */

        vector<vector<vector<double>>> dW(dense->num_out, vector<vector<double>>(dense->num_in, vector<double>(1, 0)));
        vector<double> dB(dense->num_out, 0);

        if (L == layers.size() - 2) {
          for (int i = 0; i < dense->num_out; i++) {
            double sensitivity_path_val = da_L_dz_L_per_layer[L][i][0][0];
            sensitivity_path_val *= (a[L + 1][i][0][0] - y_vector[i]);

            layer_neuron_to_sensitivity[vector<int>{L, i, 0, 0}] = sensitivity_path_val;

            for (int j = 0; j < dense->num_in; j++) {
              dW[i][j][0] = a[L - 1][j][0][0];

              dW[i][j][0] *= layer_neuron_to_sensitivity[vector<int>{L, i, 0, 0}];
              // dW[i][j][0] *= da_L_dz_L_per_layer[L][i][0][0]; // to be reused
              // dW[i][j][0] *= (a[L + 1][i][0][0] - y_vector[i]); // to be reused
            }

            dB[i] = layer_neuron_to_sensitivity[vector<int>{L, i, 0, 0}];
          }
        } else {  // runs when L = layers.size() - 4
                  /*
                  layers.size() - 1 Final activation
                  layers.size() - 2 Last Dense layer
                  layers.size() - 3 Second to last activation
                  layers.size() - 4 Second to last (Dense) layer
                  */
          for (int i = 0; i < dense->num_out; i++) {
            double sensitivity_path_val = da_L_dz_L_per_layer[L][i][0][0];

            for (int j = 0; j < dense->num_in; j++) {
              dW[i][j][0] = a[L - 1][j][0][0];

              // dW[i][j][0] *= da_L_dz_L_per_layer[L][i][0][0];

              double sum = 0;

              Dense* next_dense = dynamic_cast<Dense*>(layers[L + 2]);

              for (int k = 0; k < next_dense->num_out; k++) {
                double part_sum = next_dense->weights[k][i][0];
                part_sum *= layer_neuron_to_sensitivity[vector<int>{L + 2, k, 0, 0}];
                // part_sum *= da_L_dz_L_per_layer[L + 2][k][0][0];
                // part_sum *= (a[L + 3][k][0][0] - y_vector[k]);

                sum += part_sum;
              }
              double sensitivity_path_val_new = sensitivity_path_val * sum;

              layer_neuron_to_sensitivity[vector<int>{L, i, 0, 0}] = sensitivity_path_val_new;

              dW[i][j][0] *= layer_neuron_to_sensitivity[vector<int>{L, i, 0, 0}];
            }
            dB[i] = layer_neuron_to_sensitivity[vector<int>{L, i, 0, 0}];
          }
        }

        tuple<vector<vector<vector<double>>>, vector<double>> dParam_tuple = make_tuple(dW, dB);
        dParam_per_layer.push_back(dParam_tuple);
      }
    }
    return dParam_per_layer;
  }

  void static h_test_1(vector<vector<vector<vector<double>>>> X, int Y[100]) {
    Flatten flatten = Flatten();
    Dense dense = Dense(3, 16);
    Sigmoid sigmoid = Sigmoid();
    ConvNet model = ConvNet(vector<Layer*>{&flatten, &dense, &sigmoid});
    // Do a forward pass with the first "image"
    int label = model.predict(X[0]);
    cout << label << endl;

    if (!(label >= 0 && 10 > label)) {
      throw(string) "Test failed! " + (string) __FUNCTION__;
    }

    vector<tuple<vector<vector<vector<double>>>, vector<double>>> dParam_per_layer = model._calc_dLoss_dParam(Y[0]);
    // (L(W+h) - L(W-h))/(2*h)

    for (int i = 0; i < dense.num_out; i++) {
      for (int j = 0; j < dense.num_out; j++) {
        if (!(i == 0 && j == 0) && (rand() % 100) < 30) {
          continue;
        } else {
          tuple<vector<vector<vector<double>>>, vector<double>> dParam = dParam_per_layer[0];
          double epsilon{0.001};

          // Might be better to loop over descreasing values of epsilon
          dense.weights[i][j][0] += epsilon;
          double loss1 = model.Loss(X[0], Y[0]);

          dense.weights[i][j][0] -= 2 * epsilon;
          double loss2 = model.Loss(X[0], Y[0]);

          double num_dLoss_dWs = (loss1 - loss2) / (2 * epsilon);
          cout << get<0>(dParam)[i][j][0] << endl;
          cout << num_dLoss_dWs << endl;
          cout << "Difference in derivatives: " << num_dLoss_dWs - get<0>(dParam)[i][j][0] << endl;

          dense.weights[i][j][0] += epsilon;
        }
      }
    }
  }

  void static h_test_1_bias(vector<vector<vector<vector<double>>>> X, int Y[100]) {
    Flatten flatten = Flatten();
    Dense dense = Dense(3, 16);
    Sigmoid sigmoid = Sigmoid();
    ConvNet model = ConvNet(vector<Layer*>{&flatten, &dense, &sigmoid});
    // Do a forward pass with the first "image"
    int label = model.predict(X[0]);
    cout << label << endl;

    if (!(label >= 0 && 10 > label)) {
      throw(string) "Test failed! " + (string) __FUNCTION__;
    }

    vector<tuple<vector<vector<vector<double>>>, vector<double>>> dParam_per_layer = model._calc_dLoss_dParam(Y[0]);
    // (L(b+h) - L(b-h))/(2*h)

    for (int i = 0; i < dense.num_out; i++) {
      if (!(i == 0) && (rand() % 100) < 30) {
        continue;
      } else {
        tuple<vector<vector<vector<double>>>, vector<double>> dParam = dParam_per_layer[0];
        double epsilon{0.001};

        // Might be better to loop over descreasing values of epsilon
        dense.biases[i] += epsilon;
        double loss1 = model.Loss(X[0], Y[0]);

        dense.biases[i] -= 2 * epsilon;
        double loss2 = model.Loss(X[0], Y[0]);

        double num_dLoss_dBs = (loss1 - loss2) / (2 * epsilon);
        cout << get<1>(dParam)[i] << endl;
        cout << num_dLoss_dBs << endl;
        cout << "Difference in derivatives: " << num_dLoss_dBs - get<1>(dParam)[i] << endl;

        dense.biases[i] += epsilon;
      }
    }
  }

  void static h_test_2(vector<vector<vector<vector<double>>>> X, int Y[100]) {
    Flatten flatten = Flatten();
    Dense dense1 = Dense(8, 16);
    Sigmoid sigmoid1 = Sigmoid();
    Dense dense2 = Dense(3, 8);
    Sigmoid sigmoid2 = Sigmoid();
    ConvNet model = ConvNet(vector<Layer*>{&flatten, &dense1, &sigmoid1, &dense2, &sigmoid2});
    // Do a forward pass with the first "image"
    int label = model.predict(X[0]);
    cout << label << endl;

    if (!(label >= 0 && 10 > label)) {
      throw(string) "Test failed! " + (string) __FUNCTION__;
    }

    vector<tuple<vector<vector<vector<double>>>, vector<double>>> dParam_per_layer = model._calc_dLoss_dParam(Y[0]);
    // (L(W+h) - L(W-h))/(2*h)

    for (int i = 0; i < dense1.num_out; i++) {
      for (int j = 0; j < dense1.num_out; j++) {
        if (!(i == 0 && j == 0) && (rand() % 100) < 30) {
          continue;
        } else {
          tuple<vector<vector<vector<double>>>, vector<double>> dParam = dParam_per_layer[1];
          double epsilon{.001};

          // Might be better to loop over descreasing values of epsilon
          dense1.weights[i][j][0] += epsilon;
          double loss1 = model.Loss(X[0], Y[0]);

          dense1.weights[i][j][0] -= 2 * epsilon;
          double loss2 = model.Loss(X[0], Y[0]);

          double num_dLoss_dWs = (loss1 - loss2) / (2 * epsilon);
          cout << get<0>(dParam)[i][j][0] << endl;
          cout << num_dLoss_dWs << endl;
          cout << "Difference in derivatives: " << num_dLoss_dWs - get<0>(dParam)[i][j][0] << endl;

          dense1.weights[i][j][0] += epsilon;
        }
      }
    }
  }

  void static h_test_2_bias(vector<vector<vector<vector<double>>>> X, int Y[100]) {
    Flatten flatten = Flatten();
    Dense dense1 = Dense(8, 16);
    Sigmoid sigmoid1 = Sigmoid();
    Dense dense2 = Dense(3, 8);
    Sigmoid sigmoid2 = Sigmoid();
    ConvNet model = ConvNet(vector<Layer*>{&flatten, &dense1, &sigmoid1, &dense2, &sigmoid2});
    // Do a forward pass with the first "image"
    int label = model.predict(X[0]);
    cout << label << endl;

    if (!(label >= 0 && 10 > label)) {
      throw(string) "Test failed! " + (string) __FUNCTION__;
    }

    vector<tuple<vector<vector<vector<double>>>, vector<double>>> dParam_per_layer = model._calc_dLoss_dParam(Y[0]);
    // (L(b+h) - L(b-h))/(2*h)

    for (int i = 0; i < dense1.num_out; i++) {
      if (!(i == 0) && (rand() % 100) < 30) {
        continue;
      } else {
        tuple<vector<vector<vector<double>>>, vector<double>> dParam = dParam_per_layer[1];
        double epsilon{0.001};

        // Might be better to loop over descreasing values of epsilon
        dense1.biases[i] += epsilon;
        double loss1 = model.Loss(X[0], Y[0]);

        dense1.biases[i] -= 2 * epsilon;
        double loss2 = model.Loss(X[0], Y[0]);

        double num_dLoss_dBs = (loss1 - loss2) / (2 * epsilon);
        cout << get<1>(dParam)[i] << endl;
        cout << num_dLoss_dBs << endl;
        cout << "Difference in derivatives: " << num_dLoss_dBs - get<1>(dParam)[i] << endl;

        dense1.biases[i] += epsilon;
      }
    }
  }

  void static fit_test_1(vector<vector<vector<vector<double>>>> X, int Y[100]) {
    Flatten flatten = Flatten();
    Dense dense = Dense(3, 16);
    Sigmoid sigmoid = Sigmoid();
    ConvNet model = ConvNet(vector<Layer*>{&flatten, &dense, &sigmoid});

    model.fit(X, Y);
  }

  void static fit_test_2(vector<vector<vector<vector<double>>>> X, int Y[100]) {
    Flatten flatten = Flatten();
    Dense dense1 = Dense(8, 16);
    Sigmoid sigmoid1 = Sigmoid();
    Dense dense2 = Dense(3, 8);
    Sigmoid sigmoid2 = Sigmoid();
    ConvNet model = ConvNet(vector<Layer*>{&flatten, &dense1, &sigmoid1, &dense2, &sigmoid2});

    model.fit(X, Y);
  }
};

int main() {
  if (be_random) {
    srand(time(NULL));
  } else {
    srand(2021);
  }

  // TESTS
  // set up

  const int num_images = 100;
  vector<vector<vector<vector<double>>>> X;  // num_images x num_channels x height x width
  int Y[num_images];                         // labels for each example

  // Randomly initialize X and Y
  for (int i = 0; i < num_images; i++) {
    vector<vector<vector<double>>> image;
    vector<vector<double>> channel;  // Only one channel per image here.
    for (int j = 0; j < 4; j++) {
      vector<double> row;
      for (int k = 0; k < 4; k++) {
        double f = (double)rand() / RAND_MAX;
        double num = f;  // should be from 0 to 255 but scaled to [0, 1]
        row.push_back(num);
      }
      channel.push_back(row);
    }
    image.push_back(channel);
    X.push_back(image);
    Y[i] = rand() % 3;  // TODO: Maybe decrease number of classes for the test?
  }

  // Look at first 2 "images"
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        cout << X[i][0][j][k] << ",";
      }
      cout << endl;
    }
    cout << endl;
  }

  // tests

  try {
    // Flat convolution test
    Conv::_convolve_test();
    cout << "_convole_test done\n" << endl;

    // Depth convolution test
    Conv::convolve_test();
    cout << "convole_test done\n" << endl;

    // Flat max pool test
    MaxPool::_max_pool_test();
    cout << "_max_pool_test done\n" << endl;

    // TODO: make a depth maxpool test if necessary

    Sigmoid::sigmoid_test();
    cout << "sigmoid_test done\n" << endl;

    Relu::relu_test();
    cout << "relu_test done\n" << endl;

    Dense::h_test();
    cout << "Dense h_test done\n" << endl;

    ConvNet::h_test_1(X, Y);
    cout << "ConvNet h_test_1 done\n" << endl;

    ConvNet::h_test_1_bias(X, Y);
    cout << "ConvNet h_test_1_bias done\n" << endl;

    ConvNet::h_test_2(X, Y);
    cout << "ConvNet h_test2 done\n" << endl;

    ConvNet::h_test_2_bias(X, Y);
    cout << "ConvNet h_test_2_bias done \n" << endl;

    ConvNet::fit_test_1(X, Y);
    cout << "ConvNet fit_test_1 done \n" << endl;

    ConvNet::fit_test_2(X, Y);
    cout << "ConvNet fit_test_2 done \n" << endl;
  } catch (string my_exception) {
    cout << my_exception << endl;
    return 1;  // Do not go past the first exception in a test
  }

  cout << "Tests finished!!!!\n" << endl;

  // Main

  return 0;
}
