// Code made for finding best Hamiltonian cycle to satisfy maximise d(G) functions.
// Functions and problem described here: https://piratux.github.io/connect_points/

// best 2x2
// d(G) = 4
// 0 1 3 2 0
// [[1, 0, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 0, 1], ]
// total 0.003 ms

// best 3x3
// d(G) = 10
// 0 4 1 2 5 8 7 6 3 0
// [[1, 1, 0, 0], [1, 0, 1, 1], [2, 0, 1, 0], [2, 1, 2, 0], [2, 2, 2, 1], [1, 2, 2, 2], [0, 2, 1, 2], [0, 1, 0, 2], [0, 0, 0, 1], ]
// total 0.008 ms

// best 4x4
// d(G) = 36
// 0 6 1 2 3 10 7 11 15 9 14 13 12 5 8 4 0
// [[2, 1, 0, 0], [1, 0, 2, 1], [2, 0, 1, 0], [3, 0, 2, 0], [2, 2, 3, 0], [3, 1, 2, 2], [3, 2, 3, 1], [3, 3, 3, 2], [1, 2, 3, 3], [2, 3, 1, 2], [1, 3, 2, 3], [0, 3, 1, 3], [1, 1, 0, 3], [0, 2, 1, 1], [0, 1, 0, 2], [0, 0, 0, 1], ]
// total 4 ms

// best 5x5
// d(G) = 98
// 0 17 6 12 1 18 7 13 2 3 4 9 14 19 8 24 16 23 22 21 20 11 15 10 5 0
// [[2, 3, 0, 0], [1, 1, 2, 3], [2, 2, 1, 1], [1, 0, 2, 2], [3, 3, 1, 0], [2, 1, 3, 3], [3, 2, 2, 1], [2, 0, 3, 2], [3, 0, 2, 0], [4, 0, 3, 0], [4, 1, 4, 0], [4, 2, 4, 1], [4, 3, 4, 2], [3, 1, 4, 3], [4, 4, 3, 1], [1, 3, 4, 4], [3, 4, 1, 3], [2, 4, 3, 4], [1, 4, 2, 4], [0, 4, 1, 4], [1, 2, 0, 4], [0, 3, 1, 2], [0, 2, 0, 3], [0, 1, 0, 2], [0, 0, 0, 1], ]
// total 86 sec

// possibly best 6x6
// d(G) = 218
// [[0,0,1,0],[1,0,2,0],[2,0,3,0],[3,0,1,1],[4,0,1,1],[0,0,0,1],[0,1,0,2],[0,2,0,3],[1,2,0,3],[1,2,0,4],[2,1,0,4],[2,1,1,3],[2,2,1,3],[3,1,2,2],[3,1,0,5],[4,4,0,5],[4,4,1,5],[1,5,2,5],[2,5,3,5],[3,5,4,5],[4,5,5,5],[4,3,5,5],[4,3,5,4],[5,3,5,4],[5,2,5,3],[5,1,5,2],[5,1,3,4],[4,2,3,4],[4,2,3,3],[5,0,3,3],[5,0,2,4],[4,1,2,4],[4,1,3,2],[3,2,2,3],[2,3,1,4],[4,0,1,4]]

// possibly best 7x7
// d(G) = 426
// [[0,0,1,0],[1,0,2,0],[2,0,3,0],[3,0,4,0],[5,0,1,1],[4,0,1,1],[0,0,0,1],[0,1,0,2],[0,2,0,3],[1,2,0,3],[1,2,0,4],[2,1,0,4],[2,1,1,3],[2,2,1,3],[2,2,0,5],[3,1,0,5],[3,1,1,4],[2,3,1,4],[3,2,2,3],[4,1,3,2],[4,1,0,6],[4,5,0,6],[4,5,1,6],[5,5,1,6],[5,5,2,6],[2,6,3,6],[3,6,4,6],[4,6,5,6],[5,6,6,6],[6,5,6,6],[6,4,6,5],[6,3,6,4],[6,3,5,4],[6,2,5,4],[6,2,5,3],[5,3,4,4],[6,1,4,4],[6,1,3,5],[5,2,3,5],[5,2,4,3],[4,3,3,4],[6,0,3,4],[6,0,2,5],[5,1,2,5],[5,1,4,2],[4,2,3,3],[3,3,2,4],[2,4,1,5],[5,0,1,5]]

// possibly best 8x8
// d(G) = 768
// [[0,0,1,0],[1,0,2,0],[2,0,3,0],[3,0,4,0],[4,0,1,1],[5,0,1,1],[5,0,2,1],[6,0,2,1],[0,0,0,1],[0,1,0,2],[0,2,0,3],[1,2,0,3],[1,2,0,4],[1,3,0,4],[2,2,1,3],[2,2,0,5],[3,1,0,5],[3,1,1,4],[2,3,1,4],[3,2,2,3],[3,2,0,6],[4,1,0,6],[4,1,1,5],[2,4,1,5],[3,3,2,4],[4,2,3,3],[5,1,4,2],[5,1,0,7],[5,6,0,7],[5,6,1,7],[6,6,1,7],[6,6,2,7],[2,7,3,7],[3,7,4,7],[4,7,5,7],[5,7,6,7],[6,7,7,7],[7,6,7,7],[7,5,7,6],[7,4,7,5],[7,4,6,5],[7,3,6,5],[7,3,6,4],[6,4,5,5],[7,2,5,5],[7,2,4,6],[6,3,4,6],[6,3,5,4],[5,4,4,5],[7,1,4,5],[7,1,3,6],[6,2,3,6],[6,2,5,3],[5,3,4,4],[7,0,4,4],[7,0,3,5],[6,1,3,5],[6,1,2,6],[5,2,2,6],[5,2,4,3],[4,3,3,4],[3,4,2,5],[2,5,1,6],[6,0,1,6]]

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

#include "PiraTimer.h"

struct Point
{
    int x;
    int y;
};

struct PathInfo
{
    // We attempt to maximise d(G)
    // d(G) = |a_(i+1) - a_i|^2
    int path_length;
	
    int idx;
};

void print_array(std::vector<int> vec, int add_new_line_every_x = -1) {
    for (int i = 0; i < vec.size(); i++) {
        if (add_new_line_every_x != -1 && i % add_new_line_every_x == 0) {
            std::cout << std::endl;
        }
        std::cout << vec[i] << ' ';
    }
    std::cout << std::endl;
}

// Source: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
// To find orientation of ordered triplet (p, q, r).
// The function returns following values
// 0 --> p, q and r are collinear
// 1 --> Clockwise
// 2 --> Counterclockwise
int orientation(Point p, Point q, Point r)
{
    // See https://www.geeksforgeeks.org/orientation-3-ordered-points/
    // for details of below formula.
    int val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);

    if (val == 0) return 0;  // collinear

    return (val > 0) ? 1 : 2; // clock or counterclock wise
}

// Returns true if line segment 'p1q1' and 'p2q2' intersect (assuming they're not collinear).
bool segments_intersect(Point p1, Point q1, Point p2, Point q2)
{
    int o1 = orientation(p1, q1, p2);
    int o2 = orientation(p1, q1, q2);
    int o3 = orientation(p2, q2, p1);
    int o4 = orientation(p2, q2, q1);

    if (o1 != o2 && o3 != o4)
        return true;

    return false;
}

int make_1D_index(int x, int y, int width) {
    return y * width + x;
}

Point make_2D_index(int idx, int width) {
    Point p;
    p.x = idx % width;
    p.y = idx / width;
    return p;
}

std::vector<int> make_adjacency_matrix(int size) {
    int squared_size = size * size;

    // 0-initialise array
    std::vector<int> grid(squared_size * squared_size, 0);

    int size_m1 = size - 1;
    int idx_dest = -1;

    int64_t total_path_combinations = 1;
    std::vector<int> combination_vector;

    for (int y_from = 0; y_from < size; ++y_from) {
        for (int x_from = 0; x_from < size; ++x_from) {
            int paths_from_vertex = 0;
            for (int y_to = 0; y_to < size; ++y_to) {
                for (int x_to = 0; x_to < size; ++x_to) {
                    ++idx_dest;

                    if (x_from == x_to && y_from == y_to) {
                        continue;
                    }

                    int vec_x = std::abs(x_from - x_to);
                    int vec_y = std::abs(y_from - y_to);

                    // Remove paths that intersect with other points
                    if (std::gcd(vec_x, vec_y) > 1) {
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
                    grid[idx_dest] = squared_length;
                }
            }

            total_path_combinations *= paths_from_vertex;
            combination_vector.push_back(paths_from_vertex);
        }
    }

    return grid;
}

std::vector<int> create_lookup_table(int size) {
    int squared_size = size * size;
    std::vector<int> edge_lookup_table;
    edge_lookup_table.resize(squared_size);

    int size_m1 = size - 1;

    // Create reordered index lookup map
    // Top
    for (int i = 0; i < size_m1; i++) {
        int idx_from = i;
        int idx_to = i;
        edge_lookup_table[idx_from] = idx_to;
    }

    // Right
    for (int i = 0; i < size_m1; i++) {
        int idx_from = size_m1 + i * size;
        int idx_to = i + size_m1;
        edge_lookup_table[idx_from] = idx_to;
    }

    // Bottom
    for (int i = 0; i < size_m1; i++) {
        int idx_from = squared_size - 1 - i;
        int idx_to = i + 2 * size_m1;
        edge_lookup_table[idx_from] = idx_to;
    }

    // Left
    for (int i = 0; i < size_m1; i++) {
        int idx_from = squared_size - (i + 1) * size;
        int idx_to = i + 3 * size_m1;
        edge_lookup_table[idx_from] = idx_to;
    }

    // Remaining indices in the middle
    for (int i = 4 * size_m1; i < squared_size; i++) {
        int idx_to = i;
        int idx_offset = i - 4 * size_m1;
        Point idx_2d = make_2D_index(idx_offset, size - 2);
        int idx_from = make_1D_index(idx_2d.x + 1, idx_2d.y + 1, size);
        edge_lookup_table[idx_from] = idx_to;
    }

    return edge_lookup_table;
}

std::vector<int> create_reverse_lookup_table(int size) {
    std::vector<int> lookup_table = create_lookup_table(size);
    std::vector<int> reverse_edge_lookup_table;
    reverse_edge_lookup_table.resize(lookup_table.size());

    for (int i = 0; i < lookup_table.size(); i++) {
        int transformed_idx = lookup_table[i];
        reverse_edge_lookup_table[transformed_idx] = i;
    }

    return reverse_edge_lookup_table;
}

// 2D array of sorted connections
std::vector<PathInfo> make_path_distances(const std::vector<int>& grid, int size) {
    int squared_size = size * size;

    std::vector<PathInfo> path_distances;
    path_distances.resize(squared_size * squared_size);

    for (int i = 0; i < squared_size; i++) {
        for (int j = 0; j < squared_size; j++) {
            int idx = make_1D_index(j, i, squared_size);

            PathInfo value = { grid[idx], j };
            path_distances[idx] = value;
        }
    }

    return path_distances;
}

// Since we won't consider paths where outter grid points are visited anti-clockwise, we can just remove them
std::vector<PathInfo> prune_anti_clockwise_paths(const std::vector<PathInfo>& distances, int size) {
    int squared_size = size * size;
    int total_grid_edge_points = 4 * (size - 1);

    std::vector<PathInfo> path_distances = distances;
    std::vector<int> lookup_table = create_reverse_lookup_table(size);

    //print_array(lookup_table, size);

    for (int i = 0; i < total_grid_edge_points; i++) {
        int idx_to = lookup_table[i];
        int idx_from;
        if (i + 1 == total_grid_edge_points) {
            idx_from = lookup_table[0];
        }
        else {
            idx_from = lookup_table[i + 1];
        }

        int idx_to_prune = make_1D_index(idx_to, idx_from, squared_size);
        path_distances[idx_to_prune].path_length = 0;
    }

    return path_distances;
}

std::vector<PathInfo> sort_path_distances(const std::vector<PathInfo>& distances, int size) {
    int squared_size = size * size;

    std::vector<PathInfo> path_distances = distances;

    for (int i = 0; i < squared_size; i++) {
        // We sort the distances, so that we can iterate over available paths consecutively instead of jumping over empty indexes
        std::sort(
            path_distances.begin() + i * squared_size,
            path_distances.begin() + (i + 1) * squared_size,
            [](PathInfo left, PathInfo right) {
                return left.path_length > right.path_length;
            }
        );
    }

    return path_distances;
}

void print_coordinate_array(const std::vector<int>& solution, int size) {
    std::cout << "[";
    for (int i = 1; i < solution.size(); i++) {
        Point p1 = make_2D_index(solution[i], size);
        Point p2 = make_2D_index(solution[i - 1], size);

        std::cout << "[" << p1.x << ", " << p1.y << ", " << p2.x << ", " << p2.y << "], ";
    }

    std::cout << "]";
    std::cout << std::endl;
}

// Possible optimisations:
// - Return early if sum of additional path lengths won't exceed best path that was found (not sure if worth it)
// - If all inner grid indexes are taken and edge index was just connected, attempt to immediately calculate path length (for c(G), remaining sum would need to be pre-calculated)
// - If last inner grid index is taken that connect with next expected edge index, attempt to immediately connect it. If can't connect prune this path.
// - Split paths from middle grid points into other middle grid points and onto edges. This way we won't check indexes that we definitely won't connect with.
// - Lastly, preferable optimisation would be to somehow check if after each connection, there are no more indexes that will connect with next expected edge index without intersecting (No idea how to though).
std::vector<int> find_best_solution(const std::vector<PathInfo>& sorted_distances, int size) {
    // algorithm is undefined for these sizes
    if (size <= 1) {
        return {};
    }

    int squared_size = size * size;
    
    // 0 = free
    // 1 = taken
    std::vector<int> taken_vertex_map(squared_size, 0);

    std::vector<int> best_path_indexes;
    std::vector<int> path_indexes;
    best_path_indexes.reserve(squared_size + 1);
    path_indexes.reserve(squared_size + 1);

    // We mark first vertex as taken, since that's where we start
    taken_vertex_map[0] = 1;
    path_indexes.push_back(0);

    // Create lookup table to quickly check if vertex index is on the edge of the grid
    std::vector<int> edge_lookup_table = create_lookup_table(size);
    int total_grid_edge_points = 4 * (size - 1);

    int best_path_length = 0;

    long long total_found = 0;

    auto check_if_new_vertex_creates_intersections = [&](int new_idx, bool add_last_path) -> bool {
        // Check if there are any intersections with new path with all other ones
        Point p1 = make_2D_index(new_idx, size);
        Point p2 = make_2D_index(path_indexes.back(), size);
        for (int j = 1; j < path_indexes.size() - 1; ++j) {
            // First vertex connects with 2nd vertex and with last vertex in the path, 
            // so we know no intersection will occur there.
            if (add_last_path && j == 1) {
                continue;
            }

            // TODO: This could be optimised, because we're using one of the indexes each turn.
            Point p3 = make_2D_index(path_indexes[j], size);
            Point p4 = make_2D_index(path_indexes[j - 1], size);

            if (segments_intersect(p1, p2, p3, p4)) {
                return true;
            }
        }

        return false;
    };

    auto find_best = [&](auto& self, int last_idx, int path_length, int next_expected_grid_edge_idx) {
        // Check if last path was added
        if (path_indexes.size() == squared_size + 1) {
			total_found++;
			
			// if(total_found % 1000000 == 0){
				// std::cout << "Total paths checked: " << total_found << std::endl;
			// }
			
			// std::cout << "Total paths checked: " << total_found << std::endl;
			
			// if(total_found % 10 == 0){
				// std::cout << "Total paths checked: " << total_found << std::endl;
			// }
			
            if (path_length > best_path_length) {
                best_path_length = path_length;
                best_path_indexes = path_indexes;
				
				
				std::cout << "Total paths checked: " << total_found << std::endl;
				std::cout << "Best path length: " << best_path_length << std::endl;
				std::cout << "Best path length solution connections pairs: ";
				print_coordinate_array(path_indexes, size);
            }
			
			// std::cout << "Best path length solution connections pairs: ";
			// print_coordinate_array(path_indexes, size);

            return;
        }

        // Check if last path must be added
        bool add_last_path = false;
        if (path_indexes.size() == squared_size) {
            add_last_path = true;
        }

        // Attempt to try all paths from vertex
        for (int i = 0; i < squared_size; ++i) {
            // Check if there are still available paths
            int idx = make_1D_index(i, last_idx, squared_size);
            if (sorted_distances[idx].path_length <= 0) {
                return;
            }

            int new_idx = sorted_distances[idx].idx;
            int new_path_length = sorted_distances[idx].path_length;

            bool point_is_on_edge = false;

            // Last path must connect with the beginning
            if (add_last_path) {
                if (new_idx != 0) {
                    continue;
                }
            }
            else {
                // Each vertex must be unique, since it's Hamiltonian path
                if (taken_vertex_map[new_idx] == 1) {
                    continue;
                }

                // Optimisation: If paths divide the grid with unreachable spots, return.
                // Check if point is on edge
                int transformed_idx = edge_lookup_table[new_idx];
                if (transformed_idx < total_grid_edge_points) {
                    point_is_on_edge = true;
                }

                // Check if grid now has unreachable points
                if (point_is_on_edge && next_expected_grid_edge_idx != transformed_idx) {
                    continue;
                }
            }

            bool result = check_if_new_vertex_creates_intersections(new_idx, add_last_path);
            if (result == true) {
                continue;
            }

            path_indexes.push_back(new_idx);
            taken_vertex_map[new_idx] = 1;

            self(self, new_idx, path_length + new_path_length, next_expected_grid_edge_idx + (point_is_on_edge ? 1 : 0));
            
            path_indexes.pop_back();
            taken_vertex_map[new_idx] = 0;
        }
    };
	
	// temporary
	//path_indexes.push_back(27);
	//taken_vertex_map[27] = 1;
	//find_best(find_best, 27, 25, 1);
	
	

     find_best(find_best, 0, 0, 1);
	
	std::cout << "Total paths checked: " << total_found << std::endl;

    std::cout << "Best path length: " << best_path_length << std::endl;

    std::cout << "Best path length solution connections pairs: ";
    print_coordinate_array(best_path_indexes, size);

    std::cout << "Best path solution indexes: ";
    print_array(best_path_indexes);

    return best_path_indexes;
}

void find_best_grid_hamiltonian_cycle(int size) {
    auto grid = make_adjacency_matrix(size);
    auto distances = make_path_distances(grid, size);
    auto pruned_distances = prune_anti_clockwise_paths(distances, size);
    auto sorted_distances = sort_path_distances(pruned_distances, size);

    std::cout << "Finding best solution for " << size << "x" << size << " grid..." << std::endl;

    auto solution = find_best_solution(sorted_distances, size);

    std::cout << "All solutions checked" << std::endl;

    std::cout << std::endl;
    std::cout << std::endl;
}

int main() {
    for (int i = 3; i <= 5; i++) {
        int grid_size = i;
        PiraTimer::start("Grid size: " + std::to_string(i));
        find_best_grid_hamiltonian_cycle(grid_size);
        PiraTimer::end("Grid size: " + std::to_string(i));
    }

    PiraTimer::print_stats();
}
