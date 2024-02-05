#include <mpi.h>
#include <cstdio>
#include <vector>
#include <numeric>
#include <algorithm>

#include "PiraTimer.h"

using namespace std;

// Parallelised version of grid Hamilton cycle bruteforce search.
// Uses OpenMPI to parallelise the algorithm.
// Nodes consist of 1 master and N slaves.
// Master's job is to give tasks to slaves (in this case master sends id of starting_vertices_array where full search shall be performed).
// Slave's job is to receive task id from master, and perform bruteforce search. As well as, send master new best path it finds.
// Only master prints results, to make sure results are printed synchronously.

// This algorithm can handle 2 to N amount of slaves.
// With more slaves than tasks, tasks are distributes to slaves that are free.
// Algorithm finishes when all slaves finish their task.

// Slave amount is configured via "queue_bruteforce.sh" file by specifying -n parameter (for ex.: -n8 means there are 1 master, 7 slaves).
// Algorithm expects at least 2 nodes: 1 master and 1 slave.

// ----------------------------------------------------------------------

// Parameters:
constexpr const int n = 5; // Grid size
constexpr const int best_known_cost = 0; // Discards all paths whose cost is lower than this number, potentially finding better paths faster.

// Vertex indices that will be connected one after another starting from index 1.
// Paths with other starting vertices won't be checked.
// Example of valid starting vertices for grid size 5x5: {9, 2, 3}, which will start algorithm with edges {[1, 9], [9, 2], [2, 3]}.
// Leaving starting vertices as empty {}, will perform a full search.
// WARNING: it will not work when outter grid points are connected in anti-clockwise manner (4x4 grid invalid example: {5, 9}).
vector<vector<int>> starting_vertices_array = {
  {6},
  {7},
  {12},
  {17},
  {18}
};

// ----------------------------------------------------------------------

// Possible optimisations:
// - precalculate intersections table
// - remove symmetric edges going from vertex with index 1 (ex.: for 4x4 grid, that would remove edge (1 10) from adjacency matrix, because it's symmetric to (1 7))
// - improve cost based pruning, by only counting maximum possible distances with edges that don't intersect and with edges whose vertex isn't taken (not sure if this will improve speed).

// May as well store it here
int world_size;
int world_rank;
constexpr const int MASTER_RANK = 0;
constexpr const int n2 = n * n;

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


const int MESSAGE_TYPE_MASTER_JOB_NEW_TASK = 0;
const int MESSAGE_TYPE_MASTER_JOBS_TASKS_DEPLETED = 1;

const int MESSAGE_TYPE_SLAVE_JOB_FOUND_BEST_PATH = 0;
const int MESSAGE_TYPE_SLAVE_JOB_READY_FOR_TASK = 1;

// Preferably, we should create different packets for different message types.
// But since we're lazy, we'll just reuse this one for multiple message types.
struct CommunicationData{
    // shared data
    int message_type;
    
    // master data
    int starting_vertices_id;
    
    // slave data
    int path_indexes[n2 + 1];
    int best_cost;
};

// Source: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
// To find orientation of ordered triplet (p, q, r).
// The function returns following values
// 0 --> p, q and r are collinear
// 1 --> Clockwise
// 2 --> Counterclockwise
int orientation(Point p, Point q, Point r) {
    // See https://www.geeksforgeeks.org/orientation-3-ordered-points/
    // for details of below formula.
    int val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);

    if (val == 0) return 0;  // collinear

    return (val > 0) ? 1 : 2; // clock or counterclock wise
}

// Returns true if line segment 'p1q1' and 'p2q2' intersect (assuming they're not collinear).
// NOTE: Returns true if 2 segments share a starting/ending point
bool segments_intersect(Point p1, Point q1, Point p2, Point q2) {
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

vector<int> make_adjacency_matrix(int size) {
    int squared_size = size * size;

    // 0-initialise array
    vector<int> grid(squared_size * squared_size, 0);

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
                    grid[idx_dest] = squared_length;
                }
            }

            total_path_combinations *= paths_from_vertex;
            combination_vector.push_back(paths_from_vertex);
        }
    }

    return grid;
}

vector<int> create_lookup_table(int size) {
    int squared_size = size * size;
    vector<int> edge_lookup_table;
    edge_lookup_table.resize(squared_size);

    int size_m1 = size - 1;
    int idx_to = 0;

    // Create reordered index lookup map
    // Left
    for (int i = 0; i < size_m1; i++) {
        int idx_from = i * size;
        edge_lookup_table[idx_from] = idx_to;
        idx_to++;
    }

    // Bottom
    for (int i = 0; i < size_m1; i++) {
        int idx_from = squared_size - size_m1 + i - 1;
        edge_lookup_table[idx_from] = idx_to;
        idx_to++;
    }

    // Right
    for (int i = 0; i < size_m1; i++) {
        int idx_from = squared_size - 1 - i * size;
        edge_lookup_table[idx_from] = idx_to;
        idx_to++;
    }

    // Top
    for (int i = 0; i < size_m1; i++) {
        int idx_from = size_m1 - i;
        edge_lookup_table[idx_from] = idx_to;
        idx_to++;
    }

    // Remaining indices in the middle
    for (int i = 4 * size_m1; i < squared_size; i++) {
        int idx_offset = i - 4 * size_m1;
        Point idx_2d = make_2D_index(idx_offset, size - 2);
        int idx_from = make_1D_index(idx_2d.x + 1, idx_2d.y + 1, size);
        edge_lookup_table[idx_from] = idx_to;
        idx_to++;
    }

    return edge_lookup_table;
}

vector<int> create_reverse_lookup_table(int size) {
    vector<int> lookup_table = create_lookup_table(size);
    vector<int> reverse_edge_lookup_table;
    reverse_edge_lookup_table.resize(lookup_table.size());

    for (int i = 0; i < lookup_table.size(); i++) {
        int transformed_idx = lookup_table[i];
        reverse_edge_lookup_table[transformed_idx] = i;
    }

    return reverse_edge_lookup_table;
}

// 2D array of sorted connections
vector<PathInfo> make_path_distances(const vector<int>& grid, int size) {
    int squared_size = size * size;

    vector<PathInfo> path_distances;
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

// Since we won't consider paths where outter grid points are visited clockwise, we can just remove them
vector<PathInfo> prune_clockwise_paths(const vector<PathInfo>& distances, int size) {
    int squared_size = size * size;
    int total_grid_edge_points = 4 * (size - 1);

    vector<PathInfo> path_distances = distances;
    vector<int> lookup_table = create_reverse_lookup_table(size);

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

vector<PathInfo> prune_symmetric_paths(const vector<PathInfo>& distances, int size) {
    vector<PathInfo> path_distances = distances;
    for (int y = 0; y < size; y++) {
        for (int x = y + 1; x < size; x++) {
            int idx_to_prune = make_1D_index(x, y, size);
            path_distances[idx_to_prune].path_length = 0;
        }
    }

    return path_distances;
}

vector<PathInfo> sort_path_distances(const vector<PathInfo>& distances, int size) {
    int squared_size = size * size;

    vector<PathInfo> path_distances = distances;

    for (int i = 0; i < squared_size; i++) {
        // We sort the distances, so that we can iterate over available paths consecutively instead of jumping over empty indexes
        sort(
            path_distances.begin() + i * squared_size,
            path_distances.begin() + (i + 1) * squared_size,
            [](PathInfo left, PathInfo right) {
                return left.path_length > right.path_length;
            }
        );
    }

    return path_distances;
}

void share_best_variables(const vector<int>& path_indexes, int best_cost){
    CommunicationData data;
    for(int i = 0; i < n2 + 1; i++){
        data.path_indexes[i] = path_indexes[i];
    }
    data.best_cost = best_cost;
    data.message_type = MESSAGE_TYPE_SLAVE_JOB_FOUND_BEST_PATH;
    MPI_Send(&data, sizeof(data), MPI_BYTE, MASTER_RANK, 0, MPI_COMM_WORLD);
}

void print_best_variables(const CommunicationData& result_data, int from_rank, int job_id) {
    printf("best path found by rank %d for job id %d\n", from_rank, job_id);
    
    printf("path_indexes:\n");
    for (int i = 0; i < n2 + 1; ++i) {
        printf("%d ", result_data.path_indexes[i] + 1);
    }
    printf("\n");

    printf("path edges:\n");
    for (int i = 1; i < n2 + 1; ++i) {
        printf("%d %d\n", result_data.path_indexes[i - 1] + 1, result_data.path_indexes[i] + 1);
    }

    printf("best_cost: %d\n", result_data.best_cost);

    printf("time %lf ms\n", PiraTimer::end("bruteforce").count());
    printf("\n");
}

int find_max_possible_path_length(const vector<PathInfo>& sorted_distances, int size) {
    int squared_size = size * size;
    int max_path_length = 0;
    for (int i = 0; i < squared_size; i++) {
        max_path_length += sorted_distances[i * squared_size].path_length;
    }

    return max_path_length;
}

void find_best_solution(int size, int best_known_cost, const vector<int>& starting_vertices) {
    // algorithm is undefined for these sizes
    if (size <= 1) {
        printf("ERROR size = %d is invalid.\n", size);
        return;
    }

    vector<int> grid = make_adjacency_matrix(size);
    vector<PathInfo> distances = make_path_distances(grid, size);
    vector<PathInfo> pruned_clockwise_distances = prune_clockwise_paths(distances, size);
    vector<PathInfo> pruned_symmetric_distances = prune_symmetric_paths(pruned_clockwise_distances, size);
    vector<PathInfo> sorted_distances = sort_path_distances(pruned_symmetric_distances, size);

    int squared_size = size * size;
    int total_outter_points = (size - 1) * 4;
    
    // 0 = free
    // 1 = taken
    vector<int> taken_vertex_map(squared_size, 0);

    vector<int> best_path_indexes;
    vector<int> path_indexes;
    best_path_indexes.reserve(squared_size + 1);
    path_indexes.reserve(squared_size + 1);

    // We mark first vertex as taken, since that's where we start
    taken_vertex_map[0] = 1;
    path_indexes.push_back(0);

    // Create lookup table to quickly check if vertex index is on the edge of the grid
    vector<int> edge_lookup_table = create_lookup_table(size);
    vector<int> reverse_edge_lookup_table = create_reverse_lookup_table(size);

    int max_possible_path_length = find_max_possible_path_length(sorted_distances, size);

    int best_path_length = best_known_cost;

    long long total_found = 0;

    auto check_if_new_vertex_creates_intersections = [&](int new_idx, bool add_last_path) {
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

    auto point_is_reachable = [&](int idx_to_check, int last_outter_grid_idx, int next_outter_grid_idx) {
        bool reachable = true;

        // It's very important that when creating segment from this point to inner grid
        // point, there are no vertices that intersect with this segment, because "segments_intersect()"
        // only returns true when segments do not share starting/ending points.
        // Large prime number should make sure starting/ending points are never shared.
        Point point_outside_grid = { -1, -1009 }; // This should not cause issues for grids smaller than 1000x1000.
		
        Point current_point = make_2D_index(idx_to_check, size);
        Point p3;
        Point p4;

        // Travel backwards the edge chain, essentially checking if polygon formed via edges contains the point
        for (int i = path_indexes.size() - 1; path_indexes[i] != last_outter_grid_idx; i--) {
            p3 = make_2D_index(path_indexes[i], size);
            p4 = make_2D_index(path_indexes[i - 1], size);
            if (segments_intersect(point_outside_grid, current_point, p3, p4)) {
                reachable = !reachable;
            }
        }

        // path_indexes don't contain 2 polygon edges, so we have to check them manually
        p3 = make_2D_index(last_outter_grid_idx, size);
        p4 = make_2D_index(next_outter_grid_idx, size);
        if (segments_intersect(point_outside_grid, current_point, p3, p4)) {
            reachable = !reachable;
        }

        p3 = make_2D_index(path_indexes.back(), size);
        p4 = make_2D_index(next_outter_grid_idx, size);
        if (segments_intersect(point_outside_grid, current_point, p3, p4)) {
            reachable = !reachable;
        }

        return reachable;
    };

    auto find_best = [&](auto& self, int last_idx, int path_length, int next_expected_grid_edge_idx) {
        if (max_possible_path_length + path_length < best_path_length) {
            return;
        }

        // Check if last path was added
        if (path_indexes.size() == squared_size + 1) {
			total_found++;
			
            if (path_length > best_path_length) {
                best_path_length = path_length;
                best_path_indexes = path_indexes;

                share_best_variables(best_path_indexes, best_path_length);
            }
			
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
                if (transformed_idx < total_outter_points) {
                    point_is_on_edge = true;
                }

                if (point_is_on_edge) {
                    // Check if grid now has unreachable points
                    if (next_expected_grid_edge_idx != transformed_idx) {
                        continue;
                    }
                    
                    // Optimisation: Check if now there are unconnectable points surrounded by polygon formed by 2 outter grid points.

                    // Note: We know that if 2 adjacent outter grid points were just connected, we don't need
                    // to check for unconnectable points, since they could not have appeared.
                    if (edge_lookup_table[last_idx] + 1 != next_expected_grid_edge_idx) {
                        bool isolated_point_found = false;

                        // Iterate over inner grid points
                        for (int j = total_outter_points; j < squared_size; j++) {
                            int inner_grid_idx = reverse_edge_lookup_table[j];
                            if (taken_vertex_map[inner_grid_idx] == 0) {
                                if (point_is_reachable(inner_grid_idx, reverse_edge_lookup_table[transformed_idx - 1], new_idx) == false) {
                                    isolated_point_found = true;
                                    break;
                                }
                            }
                        }

                        if (isolated_point_found) {
                            continue;
                        }
                    }
                }
            }

            bool result = check_if_new_vertex_creates_intersections(new_idx, add_last_path);

            if (result == true) {
                continue;
            }

            path_indexes.push_back(new_idx);
            taken_vertex_map[new_idx] = 1;
            max_possible_path_length -= sorted_distances[last_idx * squared_size].path_length;

            self(self, new_idx, path_length + new_path_length, next_expected_grid_edge_idx + (point_is_on_edge ? 1 : 0));
            
            path_indexes.pop_back();
            taken_vertex_map[new_idx] = 0;
            max_possible_path_length += sorted_distances[last_idx * squared_size].path_length;
        }
    };

    // Handle starting vertices
    int last_inserted_vertex_idx = 0;
    int path_length = 0;
    int outter_grid_points = 1;
    for (int i = 0; i < starting_vertices.size(); i++) {
        int new_vertex_idx = starting_vertices[i] - 1;
        int edge_length = pruned_symmetric_distances[last_inserted_vertex_idx * squared_size + new_vertex_idx].path_length;

        if (edge_length == 0) {
            printf("ERROR Rank %d: Invalid edge connection provided in starting_vertices going from %d to %d\n", world_rank, last_inserted_vertex_idx + 1, starting_vertices[i]);
            return;
        }

        path_length += edge_length;
        max_possible_path_length -= sorted_distances[last_inserted_vertex_idx * squared_size].path_length;

        last_inserted_vertex_idx = new_vertex_idx;
        path_indexes.push_back(last_inserted_vertex_idx);
        taken_vertex_map[last_inserted_vertex_idx] = 1;

        int transformed_idx = edge_lookup_table[new_vertex_idx];
        if (transformed_idx < total_outter_points) {
            outter_grid_points++;
        }
    }

    // Finally, start the bruteforce algorithm
    find_best(find_best, last_inserted_vertex_idx, path_length, outter_grid_points);
}

void master_send_message_new_task(int slave_rank, int next_job_id){
    CommunicationData data;
    data.message_type = MESSAGE_TYPE_MASTER_JOB_NEW_TASK;
    data.starting_vertices_id = next_job_id;
    MPI_Send(&data, sizeof(data), MPI_BYTE, slave_rank, 0, MPI_COMM_WORLD);
}

void master_send_message_tasks_depleted(int slave_rank){
    CommunicationData data;
    data.message_type = MESSAGE_TYPE_MASTER_JOBS_TASKS_DEPLETED;
    MPI_Send(&data, sizeof(data), MPI_BYTE, slave_rank, 0, MPI_COMM_WORLD);
}

void slave_send_messsage_ready_for_task(){
    CommunicationData data;
    data.message_type = MESSAGE_TYPE_SLAVE_JOB_READY_FOR_TASK;
    MPI_Send(&data, sizeof(data), MPI_BYTE, MASTER_RANK, 0, MPI_COMM_WORLD);
}

int slave_receive_task_id(){
    CommunicationData data;
    MPI_Status status;
    MPI_Recv(&data, sizeof(data), MPI_BYTE, MASTER_RANK, 0, MPI_COMM_WORLD, &status);
    switch(data.message_type){
        case MESSAGE_TYPE_MASTER_JOBS_TASKS_DEPLETED:
            return -1;
            break;
        case MESSAGE_TYPE_MASTER_JOB_NEW_TASK:
            return data.starting_vertices_id;
            break;
        default:
            printf("ERROR Slave: received unknown message type\n");
            return -1;
            break;
    }
}

void master_job(){
    // Assigns tasks to slaves and collects their results to print.
    PiraTimer::start("bruteforce");
    
    int best_cost_so_far = 0;
    int next_job_id = 0;
    int total_slaves = world_size - 1;
    int slaves_informed_that_jobs_depleted = 0;
    vector<int> assigned_job_ids(total_slaves, -1);
	
	printf("Total slaves: %d\n", total_slaves);
    
    while(slaves_informed_that_jobs_depleted < total_slaves){
        CommunicationData data;
        MPI_Status status;
        MPI_Recv(&data, sizeof(data), MPI_BYTE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        
        switch(data.message_type){
            case MESSAGE_TYPE_SLAVE_JOB_READY_FOR_TASK:
                // check if there are jobs still remaining
                if(next_job_id < starting_vertices_array.size()){
                    printf("Master: Assigned job id %d to rank %d.\n", next_job_id, status.MPI_SOURCE);
                    assigned_job_ids[status.MPI_SOURCE - 1] = next_job_id;
                    master_send_message_new_task(status.MPI_SOURCE, next_job_id);
                    next_job_id++;
                }
                // all jobs depleted, tell them go home
                else{
                    master_send_message_tasks_depleted(status.MPI_SOURCE);
                    slaves_informed_that_jobs_depleted++;
                }
                break;
            case MESSAGE_TYPE_SLAVE_JOB_FOUND_BEST_PATH:
                if(data.best_cost > best_cost_so_far){
                    print_best_variables(data, status.MPI_SOURCE, assigned_job_ids[status.MPI_SOURCE - 1]);
                    best_cost_so_far = data.best_cost;
                }
                break;
            default:
                printf("ERROR Master: received unknown message type\n");
                break;
        }
    }
    
    printf("All starting vertices have been checked.\n");
    printf("total %lf ms\n", PiraTimer::end("bruteforce").count());
}

void slave_job(){
    // Performs bruteforce searches and sends results to master
    while(true){
        slave_send_messsage_ready_for_task();
        int task_id = slave_receive_task_id();
        if(task_id == -1){
            return;
        }
        
        find_best_solution(n, best_known_cost, starting_vertices_array[task_id]);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    if(world_size <= 1){
        printf("ERROR Expected at least 2 processors. Make sure to specify node count of 2 or above via parameter -n2.\n");
    }

    if (world_rank == 0) {
        master_job();
    } else {
        slave_job();
    }
    
    // Wait until all processes finish.
    MPI_Barrier(MPI_COMM_WORLD);
    
    MPI_Finalize();
}
