#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size;
    int send_data, recv_data[2];
    MPI_Request send_req[2], recv_req[2];
    MPI_Status send_status[2], recv_status[2];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    send_data = rank;  // Data to be sent

    // Determine neighboring ranks
    int left_rank = (rank - 1 + size) % size;
    int right_rank = (rank + 1) % size;

    // Send data to the left
    MPI_Isend(&send_data, 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD, &send_req[0]);
    // Receive data from the right
    MPI_Irecv(&recv_data[0], 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD, &recv_req[0]);

    // Send data to the right
    MPI_Isend(&send_data, 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD, &send_req[1]);
    // Receive data from the left
    MPI_Irecv(&recv_data[1], 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD, &recv_req[1]);

    // Wait for the completion of send and receive operations
    MPI_Waitall(2, send_req, send_status);
    MPI_Waitall(2, recv_req, recv_status);

    printf("Rank %d: Received data from left: %d, right: %d\n", rank, recv_data[0], recv_data[1]);

    MPI_Finalize();

    return 0;
}
