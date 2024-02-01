#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>

#include <chrono>
#include <cstdio>
#include <map>
#include <vector>
#include <queue>
#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>

using namespace std;

class PiraTimer
{
private:
    class Stats {
    private:
        // <total_duration, count>
        std::pair<double, size_t> averages;
        // <duration>
        std::priority_queue<double> medians;
        double calculated_median = 0.0;

    public:
        std::chrono::high_resolution_clock::time_point start_time;

    public:
        void insert_duration(double duration) {
            averages.first += duration;
            averages.second += 1;

            medians.push(duration);
        }

        // NOTE: Modifies existing median data, so can only be calculated once
        void calculate_median() {
            // Could happen when start was called, but end wasn't, or it was misspelled
            if (medians.size() == 0)
            {
                calculated_median = 0;
                return;
            }

            int64_t count = medians.size() / 2;
            double last_value = 0;
            while (count > 0) {
                last_value = medians.top();
                medians.pop();
                --count;
            }

            if (medians.size() % 2 == 0) {
                calculated_median = (last_value + medians.top()) / 2;
            }
            else {
                calculated_median = medians.top();
            }
        }

        double get_average() {
            // can happen, when timer was started, but was never ended
            if (averages.second == 0)
                return 0;

            return averages.first / (double)averages.second;
        }

        double get_total() {
            // can happen, when timer was started, but was never ended
            if (averages.second == 0)
                return 0;

            return averages.first;
        }

        double get_median() {
            return calculated_median;
        }

        size_t get_count() {
            return averages.second;
        }
    };

private:
    // Some settings
    static const int decimal_precision = 3;
    static const int spaces_between_words = 3;

    // Some duration stats that are added in end() function
    static inline std::map<std::string, Stats> stats;

public:
    static void start(const std::string& description) {
        stats[description].start_time = std::chrono::high_resolution_clock::now();
    }

    static std::chrono::duration<double, std::milli> end(const std::string& description) {
        if (stats.find(description) != stats.end()) {
            std::chrono::duration<double, std::milli> duration = std::chrono::high_resolution_clock::now() - stats[description].start_time;
            stats[description].insert_duration(duration.count());
            return duration;
        }

        // otherwise description wasn't met in "start()" function yet
        return std::chrono::duration<double, std::milli>(0);
    }

    static void print_stats() {
        std::cout << "-------------------------\n";

        auto get_num_len = [](double num) -> size_t {
            return num <= 2.0 ? 1 : log10(num) + 1;
        };

        double total_ms = 0;
        size_t max_description_length = 0;
        size_t max_average_length = 0;
        size_t max_median_length = 0;
        size_t max_total_length = 0;
        for (auto it = stats.begin(); it != stats.end(); ++it) {
            total_ms += it->second.get_average();
            it->second.calculate_median();

            max_description_length = std::max(max_description_length, it->first.length());
            max_average_length = std::max(max_average_length, get_num_len(it->second.get_average()));
            max_median_length = std::max(max_median_length, get_num_len(it->second.get_median()));
            max_total_length = std::max(max_total_length, get_num_len(it->second.get_total()));
        }

        std::cout << std::fixed << std::setprecision(decimal_precision);
        for (auto it = stats.begin(); it != stats.end(); ++it) {
            std::cout << it->first << std::string(max_description_length - it->first.length() + spaces_between_words, ' ');

            std::cout << "Average: " << it->second.get_average() << " ms" << std::string(max_average_length - get_num_len(it->second.get_average()) + spaces_between_words, ' ');
            std::cout << "Median: " << it->second.get_median() << " ms" << std::string(max_median_length - get_num_len(it->second.get_median()) + spaces_between_words, ' ');
            std::cout << "Total: " << it->second.get_total() << " ms" << std::string(max_total_length - get_num_len(it->second.get_total()) + spaces_between_words, ' ');
            std::cout << "Total measured: " << it->second.get_count() << '\n';
        }

        std::cout << "-------------------------\n";
    }

    static void reset_stats() {
        stats.clear();
    }
};

// Helper structures and functions
struct min_or_max_result {
    int idx = 0;
    int value = 0;
};

// Assumes arr is not empty
min_or_max_result array_min(const vector<int>& arr, size_t max_elements_to_walk_through) {
    min_or_max_result result;
    result.idx = 0;
    result.value = arr[0];

    for (int i = 1; i < min(arr.size(), max_elements_to_walk_through); ++i) {
        if (arr[i] < result.value) {
            result.idx = i;
            result.value = arr[i];
        }
    }

    return result;
}

// Returns index of first occurence of value.
// If element was not found, -1 is returned
// Assumes arr is not empty
int array_find(const vector<int>& arr, size_t max_elements_to_search_through, int value) {
    for (int i = 0; i < min(arr.size(), max_elements_to_search_through); i++) {
        if (arr[i] == value) {
            return i;
        }
    }

    return -1;
}

void print_variables(int total_routes, const vector<vector<int>>& edges, int best_cost, int h) {
    //if (total_routes % 1 == 0) {
    //    printf("Iterations: %d\n", total_routes);

    //    printf("edge_order:\n");
    //    for (int i = 0; i < h; ++i) {
    //        for (int j = 0; j < 2; ++j) {
    //            printf("%d ", edges[i][j] + 1);
    //        }
    //        printf("\n");
    //    }

    //    printf("current_cost: %d\n", best_cost);
    //    printf("time2 %lf ms\n", PiraTimer::end("bruteforce").count());
    //    printf("\n");
    //}
}

void print_variables2(const vector<int>& best_route, const vector<vector<int>>& edges, int best_cost, int h, int total_routes) {
    printf("best_route:\n");
    for (int i = 0; i < best_route.size(); ++i) {
        printf("%d ", best_route[i] + 1);
    }
    printf("\n");

    printf("Best edges:\n");
    for (int i = 1; i < best_route.size(); ++i) {
        printf("%d %d\n", best_route[i - 1] + 1, best_route[i] + 1);
    }
    printf("%d %d\n", best_route[best_route.size() - 1] + 1, best_route[0] + 1);

    printf("edge_order:\n");
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < 2; ++j) {
            printf("%d ", edges[i][j] + 1);
        }
        printf("\n");
    }

    printf("best_cost: %d\n", best_cost);

    printf("routes_in_total: %d\n", total_routes);
    printf("time1 %lf ms\n", PiraTimer::end("bruteforce").count());
    printf("\n");
}

void print_final_variables(int total_routes, int best_cost, int cost0, const vector<int>& best_route, int routes_in_total) {
    printf("Total number of routes: %d\n", total_routes);

    if (best_cost > cost0) {
        printf("Optimal route:\n");
        for (int i = 0; i < best_route.size(); ++i) {
            printf("%d ", best_route[i] + 1);
        }
        printf("\n");

        printf("Optimal edges:\n");
        for (int i = 1; i < best_route.size(); ++i) {
            printf("%d %d\n", best_route[i - 1] + 1, best_route[i] + 1);
        }
        printf("%d %d\n", best_route[best_route.size() - 1] + 1, best_route[0] + 1);

        printf("Optimal_cost: %d\n", best_cost);

        printf("Optimal route was: %d\n", routes_in_total);
    }
}

// intersect.m
bool intersect(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4) {
    double u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4));
    double t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4));

    double xi, yi;

    // If parameters uand v belong to interval [0, 1]
    // then (xi, yi) will be a potential cross-point of two given edges
    if ((u >= 0 && u <= 1.0) && (t >= 0 && t <= 1.0)) {
        xi = ((x3 + u * (x4 - x3)) + (x1 + t * (x2 - x1))) / 2;
        yi = ((y3 + u * (y4 - y3)) + (y1 + t * (y2 - y1))) / 2;
    } else {
        xi = NAN;
        yi = NAN;
    }

    // If the potential cross-point lies on given edges indeed, then these
    // edges intersect each other
    if (!isnan(xi)) {
        if ((min(x1, x2) <= xi && xi <= max(x1, x2) && min(y1, y2) < yi && yi < max(y1, y2)) ||
            (min(x1, x2) < xi && xi < max(x1, x2) && min(y1, y2) <= yi && yi <= max(y1, y2))) {
            return true;
        }
    }

    return false;
}

// simplify_dist.m
// modifies argument A.
void simplify_dist(
    vector<vector<int>>& A,
    const vector<vector<int>>& pind,
    const vector<int>& rows,
    const vector<int>& cols,
    int i,
    int j,
    int d
) {
    int m = d;

    int x1 = pind[i][0];
    int y1 = pind[i][1];
    int x2 = pind[j][0];
    int y2 = pind[j][1];

    for (int r = 0; r < m; ++r) {
        for (int c = 0; c < m; ++c) {
            if (A[r][c] != 0) {
                int i1 = rows[r];
                int j1 = cols[c];
                if (i1 != i && j1 != j) {
                    int x3 = pind[i1][0];
                    int y3 = pind[i1][1];
                    int x4 = pind[j1][0];
                    int y4 = pind[j1][1];
                    if (intersect(x1, y1, x2, y2, x3, y3, x4, y4)) {
                        A[r][c] = 0;
                    }
                }
            }
        }
    }
}

vector<vector<int>> gen_dist(int size) {
    int squared_size = size * size;

    // 0-initialise array
    vector<vector<int>> grid(squared_size, vector<int>(squared_size, 0));

    int size_m1 = size - 1;
    int idx_dest = -1;

    int64_t total_path_combinations = 1;
    vector<int> combination_vector;

    for (int y_from = 0; y_from < size; ++y_from) {
        for (int x_from = 0; x_from < size; ++x_from) {
            int paths_from_vertex = 0;
            for (int y_to = 0; y_to < size; ++y_to) {
                for (int x_to = 0; x_to < size; ++x_to) {
                    ++idx_dest;

                    if (x_from == x_to && y_from == y_to) {
                        continue;
                    }

                    int vec_x = abs(x_from - x_to);
                    int vec_y = abs(y_from - y_to);

                    // Remove paths that intersect with other points
                    if (gcd(vec_x, vec_y) > 1) {
                        continue;
                    }

                    // Remove paths that split the grid where no Hamiltonian cycle can be constructed
                    // Here, {W, N, E, S} indicate grid points that are on the edge of the grid
                    //     N
                    //    ---
                    // W |   | E
                    //    ---
                    //     S
                    if (size > 2) {
                        // W-E paths
                        if (vec_x == size_m1) {
                            continue;
                        }

                        // N-S paths
                        if (vec_y == size_m1) {
                            continue;
                        }

                        // Only check for connections that don't touch corners
                        if (vec_x >= 1 && vec_y >= 1) {
                            // N-W paths
                            if (x_from == 0 && y_to == 0 ||
                                y_from == 0 && x_to == 0) {
                                continue;
                            }

                            // S-E paths
                            if (x_from == size_m1 && y_to == size_m1 ||
                                y_from == size_m1 && x_to == size_m1) {
                                continue;
                            }

                            // N-E paths
                            if (x_from == size_m1 && y_to == 0 ||
                                y_from == 0 && x_to == size_m1) {
                                continue;
                            }

                            // S-W paths
                            if (x_from == 0 && y_to == size_m1 ||
                                y_from == size_m1 && x_to == 0) {
                                continue;
                            }
                        }
                    }

                    paths_from_vertex += 1;

                    int squared_length = vec_x * vec_x + vec_y * vec_y;
                    grid[idx_dest % squared_size][idx_dest / squared_size] = squared_length;
                }
            }

            total_path_combinations *= paths_from_vertex;
            combination_vector.push_back(paths_from_vertex);
        }
    }

    return grid;
}

// gen_dist.m
//vector<vector<int>> gen_dist(int n, const vector<vector<int>>& pnum, const vector<vector<int>>& pind) {
//    int m = n * n;
//    vector<vector<int>> A(m, vector<int>(m, 0));
//
//    // At first we calculate Euclidean distance squares from each vertex of the grid to each vertex.
//    for (int i = 0; i < m; ++i) {
//        for (int j = i + 1; j < m; ++j) {
//            int x1 = pind[i][0];
//            int y1 = pind[i][1];
//            int x2 = pind[j][0];
//            int y2 = pind[j][1];
//            A[i][j] = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
//            A[j][i] = A[i][j];
//        }
//    }
//
//    // Then we change distance A(i,j)=0 if vertices i and j both belong to
//    // boundaries of the grid
//    for (int i = 0; i < m; ++i) {
//        for (int j = i + 1; j < m; ++j) {
//            int x1 = pind[i][0];
//            int y1 = pind[i][1];
//            int x2 = pind[j][0];
//            int y2 = pind[j][1];
//            int edge = (x1 == 1) + (x1 == n) + (y1 == 1) + (y1 == n) + (x2 == 1) + (x2 == n) + (y2 == 1) + (y2 == n);
//            if (edge > 1 && !(x1 == y1 && edge == 2) && !(x2 == y2 && edge == 2) && !(x1 + y1 == n + 1 && edge == 2) && !(x2 + y2 == n + 1 && edge == 2)) {
//                A[i][j] = 0;
//                A[j][i] = 0;
//            }
//            if (x1 == x2 && abs(y1 - y2) > 1) {
//                A[i][j] = 0;
//                A[j][i] = 0;
//            }
//            if (y1 == y2 && abs(x1 - x2) > 1) {
//                A[i][j] = 0;
//                A[j][i] = 0;
//            }
//        }
//    }
//
//    // Now we have to restore the distances for some pairs of vertices that
//    // belong to the same boundary
//    for (int i = 0; i < m; ++i) {
//        for (int dx = -1; dx <= 1; dx += 2) {
//            int x = pind[i][0] + dx;
//            int y = pind[i][1];
//            if (0 <= x && x < n) {
//                int j = pnum[x][y];
//                A[i][j] = 1;
//            }
//        }
//        for (int dy = -1; dy <= 1; dy += 2) {
//            int y = pind[i][1] + dy;
//            int x = pind[i][0];
//            if (0 <= y && y < n) {
//                int j = pnum[x][y];
//                A[i][j] = 1;
//            }
//        }
//    }
//
//    // Finally, we consider all diagonals as well as horizontals and verticals
//    // and change distance to zero if two vertices are not seen from each other in the grid
//    // (straight line connecting them crosses other vertices).
//    for (int dx = -(n - 2); dx <= (n - 2); ++dx) {
//        for (int dy = -(n - 2); dy <= (n - 2); ++dy) {
//            for (int i = 0; i < m; ++i) {
//                if (dx != 0 && dy != 0) {
//                    int x0 = pind[i][0] + 1;
//                    int y0 = pind[i][1] + 1;
//                    int kmax = floor((double)abs(x0 - 1) / (double)abs(dx));
//                    int lmax = floor((double)abs(y0 - 1) / (double)abs(dy));
//                    int kiek = min(kmax, lmax);
//                    if (kiek > 1) {
//                        for (int k = 2; k <= kiek; ++k) {
//                            int x = x0 - k * dx;
//                            int y = y0 - k * dy;
//                            int j = pnum[x - 1][y - 1];
//                            A[i][j] = 0;
//                        }
//                    }
//                    kmax = floor((double)abs(x0 - 1) / (double)abs(dx));
//                    lmax = floor((double)abs(y0 - n) / (double)abs(dy));
//                    kiek = min(kmax, lmax);
//                    if (kiek > 1) {
//                        for (int k = 2; k <= kiek; ++k) {
//                            int x = x0 - k * dx;
//                            int y = y0 + k * dy;
//                            int j = pnum[x - 1][y - 1];
//                            A[i][j] = 0;
//                        }
//                    }
//                    kmax = floor((double)abs(x0 - n) / (double)abs(dx));
//                    lmax = floor((double)abs(y0 - 1) / (double)abs(dy));
//                    kiek = min(kmax, lmax);
//                    if (kiek > 1) {
//                        for (int k = 2; k <= kiek; ++k) {
//                            int x = x0 + k * dx;
//                            int y = y0 - k * dy;
//                            int j = pnum[x - 1][y - 1];
//                            A[i][j] = 0;
//                        }
//                    }
//                    kmax = floor((double)abs(x0 - n) / (double)abs(dx));
//                    lmax = floor((double)abs(y0 - n) / (double)abs(dy));
//                    kiek = min(kmax, lmax);
//                    if (kiek > 1) {
//                        for (int k = 2; k <= kiek; ++k) {
//                            int x = x0 + k * dx;
//                            int y = y0 + k * dy;
//                            int j = pnum[x - 1][y - 1];
//                            A[i][j] = 0;
//                        }
//                    }
//                }
//            }
//        }
//    }
//
//    return A;
//}

// Hamilton4n.m
void solveMaxTSP(const int n) {
    // Initialization to solve maxTSP -- find maximal weight Hamilton cycle
    // Grid nxn with n2 = n ^ 2 vertices, t0 to measure program execution time
    // 
    // distances - initial n ^ 2 x n ^ 2 matrix of distances in grid.
    // Routes cannot include edges(u, v) where(a) u and v lie on different boundaries of
    // the grid, or (b)straight line uv crosses another vertex w in our grid.
    // In such cases dist(u, v) = 0.
    // A - current matrix of distances of size d x d.Initially d = n ^ 2 or d = m,
    // where m <= n2 will be defined below.
    // Arrays rows and cols show numbers of remained rows and columns in current matrix A
    // 
    // edges - list of edges(edge = pair of vertices) in current route
    // dist - length of current route
    const int n2 = n * n; // #VALUE
    vector<int> rows(n2); // #INDEXES_FROM_0
    vector<int> cols(n2); // #INDEXES_FROM_0
    for (int i = 0; i < n2; i++) {
        rows[i] = i;
        cols[i] = i;
    }
    
    int dist = 0; // #VALUE
    vector<vector<int>> edges(n2, vector<int>(2, 0)); // #INDEXES_FROM_0

    // Assign numbers from 1 to n^2 to grid vertices and store them in nxn array pnum
    // We consider grid nxn as a matrix and enumerate its vertices like matrix
    // elements starting from upper left corner.
    // Inverse to pnum n^2 x 2 array pind stores row and column numbers [i, j] of vertex with number k
    // We need these coordinates [i, j] to verify if two straight edges intersect in
    // Euclidean plane
    vector<vector<int>> pnum(n, vector<int>(n)); // #INDEXES_FROM_0
    vector<vector<int>> pind(n2, vector<int>(2)); // #INDEXES_FROM_0
    int k = 0; // #INDEXES_FROM_0

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            pnum[i][j] = k;
            pind[k][0] = i;
            pind[k][1] = j;
            k = k + 1;
        }
    }

    vector<vector<int>> distances = gen_dist(n); // n^2 x n^2 matrix  // #VALUE
    vector<vector<int>> A = distances; // #VALUE

    // Initialization of current paths starting from each vertex k.
    // Initially each path consists of one vertex k
    // and its length is equal to 1.
    // n2 x 1 array ends stores the last vertex of each path.
    vector<vector<int>> paths(n2, vector<int>(n2, 0)); // #INDEXES_FROM_0
    for (int i = 0; i < n2; ++i) {
        paths[i][0] = i;
    }
    vector<int> lengths(n2, 1); // #INDEXES_FROM_1
    vector<int> ends(n2, -1); // #INDEXES_FROM_0
    vector<int> best_route(n2, 0); // #INDEXES_FROM_0
    int cost0 = 0; // #VALUE (UNUSED?)
    int best_cost = cost0; // #VALUE
    int total_routes = 0; // #VALUE

    // For backtracking we need to remember current distance matrix,
    // current rowand column numbers, current pathsand cost of current partial
    // route for each search depth h = 0, ..., (n2 - 1). When we return from depth h to h - 1 we
    // use stored values to update current data.
    vector<vector<vector<int>>> matrices(n2, vector<vector<int>>(n2, vector<int>(n2, 0)));
    vector<vector<int>> all_rows(n2, vector<int>(n2, 0));
    vector<vector<int>> all_cols(n2, vector<int>(n2, 0));
    vector<vector<vector<int>>> all_paths(n2, vector<vector<int>>(n2, vector<int>(n2, 0)));
    vector<vector<int>> all_lengths(n2, vector<int>(n2, 0));
    vector<vector<int>> all_ends(n2, vector<int>(n2, 0));
    vector<int> costs(n2, 0);

    // If we start from nonempty partial route we have to initialize
    // current paths, current nonused rows and columns and
    // current distance matrix in program code manually.
    // c0 is cost of current route.
    // Current m x m distance matrix A usually is inserted by Copy-Paste from Ham_intro
    // program's screen. In this particular program Hamilton4 we start from
    // empty route, so initially A is of order d=m=n^2. Later its order depends on
    // depth h.
    int c0 = 0; // #VALUE (UNUSED?)
    int m = n2; // #VALUE

    // Start of backtracking.
    // h = -1 will mean that all possible routes were checked.
    int h = 0; // #INDEXES_FROM_0

    // Undeclared variables found used in algorithm
    int d = 0; // #INDEXES_FROM_1
    int factor = 0; // #VALUE
    int rmin = 0; // #VALUE
    int rind = 0; // #INDEXES_FROM_0
    int cmin = 0; // #VALUE
    int cind = 0; // #INDEXES_FROM_0
    int c = 0; // #INDEXES_FROM_0
    int r = 0; // #INDEXES_FROM_0
    int a1 = 0; // #INDEXES_FROM_0
    int a2 = 0; // #INDEXES_FROM_0
    int ma1 = 0; // #VALUE
    int ma2 = 0; // #VALUE
    vector<vector<int>> Anew(m, vector<int>(m, 0));
    vector<int> opposite(2, 0); // #INDEXES_FROM_0
    int routes_in_total = 0; // #VALUE
    vector<int> temp_arr(m, 0); // Used when copying arrays to avoid overlap issue
    vector<vector<int>> A_last(m, vector<int>(m, 0));
    vector<vector<int>> A_old(m, vector<int>(m, 0));
    vector<int> M(m, 0);

    vector<int> srow(m, 0);
    vector<int> scol(m, 0);

    auto print_variables_to_file = [&]() {
        return;
        if (total_routes > 200) {
            exit(1);
            //return;
        }

        ofstream outfile("output.txt", ios::app); // Open file in append mode, change "output.txt" to your desired file name

        if (!outfile.is_open()) {
            cerr << "Error opening the file!" << endl;
            return;
        }


        outfile << endl;
        outfile << "iterations =" << endl << endl;
        outfile << "\t" << total_routes << endl << endl;


        outfile << endl;
        outfile << "A =" << endl << endl;
        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j) {
                outfile << "\t" << A[i][j];
            }
            outfile << endl;
        }
        outfile << endl;


        outfile << endl;
        outfile << "edge_order =" << endl << endl;
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < 2; ++j) {
                outfile << "\t" << edges[i][j] + 1;
            }
            outfile << endl;
        }
        outfile << endl;

        outfile.close();
    };

    auto calculate_min_and_factor = [&]() {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                srow[i] = 0;
                scol[i] = 0;
            }
        }

        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j) {
                srow[i] += (A[i][j] > 0) ? 1 : 0;
                scol[i] += (A[j][i] > 0) ? 1 : 0;
            }
        }

        auto _row_min = array_min(srow, d);
        auto _col_min = array_min(scol, d);

        rmin = _row_min.value;
        rind = _row_min.idx;
        cmin = _col_min.value;
        cind = _col_min.idx;
        factor = min(rmin, cmin);
    };

    while (h > -1) {
        // d - order of current distance matrix A (number of unvisited vertices)
        // factor - logical key that shows if current route is still acceptable
        d = m - h;
        factor = 1;

        while (factor > 0) {
            calculate_min_and_factor();

            //if (total_routes == 134) {
            //    printf("");
            //}

            if (factor == 0) {
                // We found a row or column with all zeros, so current partial
                // route cannot be prolonged, number of finished partial routes
                // increases by 1.
                // Instead of finishing current iteration, we move
                // to next iteration of the most inner cycle(instruction "continue").
                total_routes++;
                print_variables(total_routes, edges, best_cost, h);
                print_variables_to_file();
                continue;
            }

            // This line is reached if factor=1, i.e. there exists at least one
            // edge acceptable to prolong current route. We select the longest
            // edge [a1 a2] and replace its length in matrix A by 0.
            if (rmin <= cmin) {
                r = rind;
                a1 = rows[rind];

                // [ma2,c]=max(A(rind,1:d));
                c = 0;
                ma2 = A[rind][0];
                for (int i = 1; i < d; ++i) {
                    if (A[rind][i] > ma2) {
                        c = i;
                        ma2 = A[rind][i];
                    }
                }

                a2 = cols[c];
                A[rind][c] = 0;
            }
            else {
                c = cind;
                a2 = cols[cind];

                // [ma1, r] = max(A(1:d, cind));
                r = 0;
                ma1 = A[0][cind];
                for (int i = 1; i < d; ++i) {
                    if (A[i][cind] > ma1) {
                        r = i;
                        ma1 = A[i][cind];
                    }
                }

                a1 = rows[r];
                A[r][cind] = 0;
            }

            // Now we add selected edge to list of edges in current route and
            // recalculate the length of the route.
            edges[h][0] = a1;
            edges[h][1] = a2;
            dist = costs[h] + distances[a1][a2];

            // Store all current data to use for another edge choice in depth h
            // A(1:d,d+1:m)=0;
            for (int i = 0; i < d; ++i) {
                for (int j = d; j < m; ++j) {
                    A[i][j] = 0;
                }
            }
            // A(d+1:n2,1:m)=0;
            for (int i = d; i < n2; ++i) {
                for (int j = 0; j < m; ++j) {
                    A[i][j] = 0;
                }
            }

            matrices[h] = A;
            all_rows[h] = rows;
            all_cols[h] = cols;
            all_paths[h] = paths;
            all_lengths[h] = lengths;
            all_ends[h] = ends;

            // If current route is not complete we increase the depth hand move deeper with smaller matrix
            if (d > 1) {
                h = h + 1;
                d = d - 1;
                costs[h] = dist;
                for (int i = 0; i < r; ++i) {
                    for (int j = 0; j < d + 1; ++j) {
                        Anew[i][j] = A[i][j];
                    }
                }

                for (int i = r; i < d; ++i) {
                    for (int j = 0; j < d + 1; ++j) {
                        Anew[i][j] = A[i + 1][j];
                    }
                }

                for (int j = c; j < d; ++j) {
                    for (int i = 0; i < d; ++i) {
                        Anew[i][j] = Anew[i][j + 1];
                    }
                }

                for (int i = 0; i < d; ++i) {
                    for (int j = 0; j < d; ++j) {
                        A[i][j] = Anew[i][j];
                    }
                }

                // Recalculate remaining rows and columns
                int loc = 0;
                loc = array_find(rows, rows.size(), a1);
                for (int i = loc; i < d; i++) {
                    rows[i] = rows[i + 1];
                }

                loc = array_find(cols, cols.size(), a2);
                for (int i = loc; i < d; i++) {
                    cols[i] = cols[i + 1];
                }

                // Most complicated part: we have to recalculate paths to include
                // new edge [a1 a2]. We also have to exclude appearance of shorter cycles. For
                // this reason after finding path P that includes new edge we
                // interdict to connect an end of P with its starting vertex.
                // An edge to interdict is denoted by "opposite".

                loc = array_find(ends, ends.size(), a1);
                if (loc == -1) {
                    if (lengths[a2] > 1) {
                        // Case 1: There is no path that ends in a1 but there is a path P starting in a2 --
                        // we need to add edge [a1 a2] to the front of P.
                        for (int i = 0; i < lengths[a2]; i++) {
                            temp_arr[i] = paths[a2][i];
                        }
                        for (int i = 0; i < lengths[a2]; i++) {
                            paths[a1][i + 1] = temp_arr[i];
                        }
                        lengths[a1] = lengths[a2] + 1;
                        ends[a1] = ends[a2];

                        // If new path is a cycle, then current partial route
                        // cannot be prolonged, go to next iteration.
                        if (a1 == ends[a1]) {
                            continue;
                        }

                        ends[a2] = -1;
                        opposite[0] = ends[a1];
                        opposite[1] = a1;
                    }
                    else {
                        // Case 2: There is no path that ends in a1 and there is no path starting in a2 --
                        // we have to create new path equal to one edge [a1 a2].
                        paths[a1][1] = a2;
                        lengths[a1] = 2;
                        ends[a1] = a2;
                        if (a1 == ends[a1]) {
                            continue;
                        }
                        opposite[0] = a2;
                        opposite[1] = a1;
                    }
                }
                else {
                    // Case 3: There is a (unique) path that ends in a1. It starts in i1. We add edge [a1 a2] to its end.              
                    int i1 = loc;
                    lengths[i1] = lengths[i1] + 1;
                    paths[i1][lengths[i1] - 1] = a2;
                    ends[i1] = a2;
                    if (i1 == ends[i1]) {
                        continue;
                    }
                    opposite[0] = a2;
                    opposite[1] = i1;

                    // Case 4: There is a path that ends in a1 and there is a path starting in a2 --
                    // we have to connect two old paths into one new path (connect them with an edge [a1 a2]).
                    if (lengths[a2] > 1) {
                        for (int i = 0; i < lengths[a2]; i++) {
                            temp_arr[i] = paths[a2][i];
                        }
                        for (int i = 0; i < lengths[a2]; i++) {
                            paths[i1][i + lengths[i1] - 1] = temp_arr[i];
                        }
                        lengths[i1] = lengths[i1] + lengths[a2] - 1;
                        ends[i1] = ends[a2];
                        if (i1 == ends[i1]) {
                            continue;
                        }
                        ends[a2] = -1;
                        opposite[0] = ends[i1];
                        opposite[1] = i1;
                    }
                }

                // Finally, we store recalculated paths.
                all_paths[h] = paths;
                all_lengths[h] = lengths;
                all_ends[h] = ends;

                // For a moment we store current matrix A
                for (int i = 0; i < d; i++) {
                    for (int j = 0; j < d; j++) {
                        A_last[i][j] = A[i][j];
                    }
                }

                // We check if both ends of new path belong to remaining rows
                // and columns. If yes, we interdict an edge between them.
                int B = 0;
                int C = 0;
                for (int i = 0; i < d; i++) {
                    if (rows[i] == opposite[0]) {
                        B++;
                    }
                    if (cols[i] == opposite[1]) {
                        C++;
                    }
                }
                if (B + C == 2) {
                    int i = array_find(rows, d, opposite[0]);
                    int j = array_find(cols, d, opposite[1]);
                    A[i][j] = 0;
                }

                // Now we have to recalculate remaining distances after adding
                // the new edge [a1 a2].
                // But if the matrix A was already 1x1, the "interdicted" cycle was
                // actually already Hamilton cycle. In this case
                // our previuos change of A(i, j) was incorrect.
                // Therefore, depending on the case, we send two different versions of matrix A
                // to recalculate all distances after adding to the route a new edge
                for (int i = 0; i < d; i++) {
                    for (int j = 0; j < d; j++) {
                        A_old[i][j] = A[i][j];
                    }
                }
                if (d == 1) {
                    A_old = A_last;
                }

                simplify_dist(A_old, pind, rows, cols, a1, a2, d);
                for (int i = 0; i < d; i++) {
                    for (int j = 0; j < d; j++) {
                        A[i][j] = A_old[i][j];
                    }
                }

                // After simplification it could appear that for some vertex in a grid
                // there are no more incoming or outgoing edges. We check and go
                // to new iteration in such case.
                calculate_min_and_factor();
                if (factor == 0) {
                    total_routes++;
                    print_variables(total_routes, edges, best_cost, h);
                    print_variables_to_file();
                    continue;
                }

                // In case of 2x2 matrix it could appear that after removing row
                // and column only two vertices of grid are left but they cannot be
                // connected. It means that current route cannot be prolonged --
                // we go to new iteration.
                if (d == 1 && A[0][0] == 0) {
                    total_routes++;
                    print_variables(total_routes, edges, best_cost, h);
                    print_variables_to_file();
                    continue;
                }

                // After simplification it could appear that there are no more edges long enough
                // to receive a better solution than we had found before.
                // We checkand go to new iteration in such case.

                // A1=A(1:d,1:d);
                // S=max(sum(max(A1)),sum(max(A1')));
                int sum1 = 0;
                int sum2 = 0;
                for (int i = 0; i < d; i++) {
                    int max1 = 0;
                    int max2 = 0;
                    for (int j = 0; j < d; j++) {
                        if (A[i][j] > max1) {
                            max1 = A[i][j];
                        }
                        if (A[j][i] > max2) {
                            max2 = A[j][i];
                        }
                    }
                    sum1 += max1;
                    sum2 += max2;
                }
                int S = max(sum1, sum2);

                if (dist + S + c0 < best_cost) {
                    total_routes++;
                    print_variables(total_routes, edges, best_cost, h);
                    print_variables_to_file();
                    continue;
                }
            }

            // That was the end of case d >= 2, i.e. if matrix A was at least 2x2.

            // The case when two vertices of the grid are left
            // and still there is an edge connecting them -- a new Hamilton cycle is found. 
            // It remains to check if the new route is better than the best found before.
            if (d == 1 && A[0][0] > 0){
                total_routes++;
                print_variables(total_routes, edges, best_cost, h);
                print_variables_to_file();

                for (int i = 0; i < paths[opposite[1]].size(); i++) {
                    M[i] = paths[opposite[1]][i];
                }
                dist = dist + distances[opposite[0]][opposite[1]] + c0;
                if (dist > best_cost) {
                    for (int i = 0; i < M.size(); i++) {
                        best_route[i] = M[i];
                    }
                    best_cost = dist;
                    routes_in_total = total_routes;
                    print_variables2(best_route, edges, best_cost, h, total_routes);
                }
            }
        }

        // If all possible edges in depth h were considered, we have to return
        // back to depth h-1 and choose a new edge in the previous depth
        h = h - 1;

        // If the previous depth was not zero, we have to restore all data of this
        // depth and continue the search.
        if (h > -1) {
            A = matrices[h];
            rows = all_rows[h];
            cols = all_cols[h];
            paths = all_paths[h];
            lengths = all_lengths[h];
            ends = all_ends[h];
            dist = costs[h];
        }
        // If previous depth was -1, we display final results and stop 
        else {
            print_final_variables(total_routes, best_cost, cost0, best_route, routes_in_total);
        }
    }
}

int main() {
    // Clear contents of output.txt
    ofstream clearFile("output.txt", ios::trunc);
    clearFile.close();

    PiraTimer::start("bruteforce");

    int n = 5;
    solveMaxTSP(n);

    PiraTimer::end("bruteforce");
    PiraTimer::print_stats();
}
