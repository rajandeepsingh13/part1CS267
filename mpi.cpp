#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#include <iostream>

#include <array>
#include <vector>

#include "common.h"

typedef struct
{
  std::vector<std::array<int, 2>> id;
  std::vector<particle_t> particle;
  std::vector<std::array<int, 2>> neighbor_bin_id;
  //int nparticle;
} bin_t;

typedef struct
{
  particle_t particle;
  int bin_idx;
  int bin_idy;
} particle_buffer1;

//
//  benchmarking program
//
int main( int argc, char **argv )
{    
    int navg, nabsavg=0;
    double dmin, absmin=1.0,davg,absavg=0.0;
    double rdavg,rdmin;
    int rnavg; 
 
    //
    //  process command line parameters
    //
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        printf( "-no turns off all correctness checks and particle output\n");
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );
    
    //
    //  set up MPI
    //
    int n_proc, rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    
    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;


    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    
    // Define new MPI datatype
    MPI_Datatype PARTICLE;
    MPI_Type_contiguous( 6, MPI_DOUBLE, &PARTICLE );
    MPI_Type_commit( &PARTICLE );

    MPI_Datatype types[2] = {MPI_DOUBLE, MPI_INT};
    MPI_Datatype PBUFFER1;
    int block_length[2] = {6, 2};
    MPI_Aint offsets[2] = {0, 6*sizeof(double)};
    MPI_Type_create_struct(2, block_length, offsets, types, &PBUFFER1);
    MPI_Type_commit( &PBUFFER1 );  


    // Initialize bin and assign neighboring bins
    double bin_size = cutoff;
    double area_size = sqrt( density * n );
    // Calculate # of bins in each direction
    int nbin_1d = floor(area_size/bin_size);
    if(fmod(area_size, bin_size) != 0) nbin_1d++;
    // Initialize 2D array of the bin, the 3rd dimension is just padding to prevent false sharing
    bin_t bin[nbin_1d][nbin_1d];
    // Assign neighboring bin id to each bin
    for(int row = 0; row < nbin_1d; row++) {
      for(int col = 0; col < nbin_1d; col++) {
        std::array<int, 2> id_temp = {row, col};
        bin[row][col].id.push_back(id_temp);
        if(row == 0) {
	  if(col == 0) {
            std::array<int, 2> temp[3];
            temp[0] = {0,1};
            temp[1] = {1,0};
            temp[2] = {1,1};
            for(int i = 0; i < sizeof(temp)/sizeof(*temp); i++)
	      bin[row][col].neighbor_bin_id.push_back(temp[i]);
	  } else if (col == nbin_1d-1) {
	    std::array<int, 2> temp[3];
            temp[0] = {row+1,col};
            temp[1] = {row,col-1};
            temp[2] = {row+1,col-1};
            for(int i = 0; i < sizeof(temp)/sizeof(*temp); i++)
	      bin[row][col].neighbor_bin_id.push_back(temp[i]);
    	  } else {
	    std::array<int, 2> temp[5];
            temp[0] = {row,col-1};
            temp[1] = {row,col+1};
            temp[2] = {row+1,col-1};
            temp[3] = {row+1,col};
            temp[4] = {row+1,col+1};
            for(int i = 0; i < sizeof(temp)/sizeof(*temp); i++)
	      bin[row][col].neighbor_bin_id.push_back(temp[i]);
    	  } 
	}    
        else if(row == nbin_1d-1) {
	  if(col == 0) {
            std::array<int, 2> temp[3];
            temp[0] = {row-1,0};
            temp[1] = {row-1,1};
            temp[2] = {row,1};
            for(int i = 0; i < sizeof(temp)/sizeof(*temp); i++)
	      bin[row][col].neighbor_bin_id.push_back(temp[i]);
	  } else if (col == nbin_1d-1) {
	    std::array<int, 2> temp[3];
            temp[0] = {row-1,col};
            temp[1] = {row-1,col-1};
            temp[2] = {row,col-1};
            for(int i = 0; i < sizeof(temp)/sizeof(*temp); i++)
	      bin[row][col].neighbor_bin_id.push_back(temp[i]);
    	  } else {
	    std::array<int, 2> temp[5];
            temp[0] = {row,col-1};
            temp[1] = {row,col+1};
            temp[2] = {row-1,col-1};
            temp[3] = {row-1,col};
            temp[4] = {row-1,col+1};
            for(int i = 0; i < sizeof(temp)/sizeof(*temp); i++)
	      bin[row][col].neighbor_bin_id.push_back(temp[i]);
    	  }
	}
	else {
	  if(col == 0) {
            std::array<int, 2> temp[5];
            temp[0] = {row-1,0};
            temp[1] = {row-1,1};
            temp[2] = {row,1};
            temp[3] = {row+1,0};
            temp[4] = {row+1,1};
            for(int i = 0; i < sizeof(temp)/sizeof(*temp); i++)
	      bin[row][col].neighbor_bin_id.push_back(temp[i]);
	  } else if (col == nbin_1d-1) {
	    std::array<int, 2> temp[5];
            temp[0] = {row-1,col};
            temp[1] = {row-1,col-1};
            temp[2] = {row,col-1};
            temp[3] = {row+1,col-1};
            temp[4] = {row+1,col};
            for(int i = 0; i < sizeof(temp)/sizeof(*temp); i++)
	      bin[row][col].neighbor_bin_id.push_back(temp[i]);
    	  } else {
	    std::array<int, 2> temp[8];
            temp[0] = {row,col-1};
            temp[1] = {row,col+1};
            temp[2] = {row-1,col-1};
            temp[3] = {row-1,col};
            temp[4] = {row-1,col+1};
            temp[5] = {row+1,col-1};
            temp[6] = {row+1,col};
            temp[7] = {row+1,col+1};
            for(int i = 0; i < sizeof(temp)/sizeof(*temp); i++)
	      bin[row][col].neighbor_bin_id.push_back(temp[i]);
    	  }
	}
      }
    }
   
    //
    //  initialize and distribute the particles (that's fine to leave it unoptimized)
    //
    set_size( n );
    if( rank == 0 )
        init_particles( n, particles );
    MPI_Bcast(particles, n, PARTICLE, 0, MPI_COMM_WORLD);
 
    // Partition of bins
    int nbinrow_pp = floor(nbin_1d/n_proc);
    int residual_b = nbin_1d%n_proc;
    int nbinrow[n_proc];
    for(int i = 0; i < n_proc; i++) {
      nbinrow[i] = nbinrow_pp;
      if(i < residual_b) nbinrow[i]++;
    } 
    int nbinrow_before[n_proc];
    memset(nbinrow_before, 0, n_proc*sizeof(int));
    for(int i = 0; i < n_proc; i++) {
      for(int j = 0; j < i; j++) {
	nbinrow_before[i] += nbinrow[j];
      }
    }

    // Partition of particles
    int nparticle_pp = floor(n/n_proc);
    int residual_p = n%n_proc;
    int nparticle[n_proc];
    for(int i = 0; i < n_proc; i++) {
      nparticle[i] = nparticle_pp;
      if(i < residual_p) nparticle[i]++;
    } 
    int nparticle_before[n_proc];
    memset(nparticle_before, 0, n_proc*sizeof(int));
    for(int i = 0; i < n_proc; i++) {
      for(int j = 0; j < i; j++) {
	nparticle_before[i] += nparticle[j];
      }
    }
 
    std::vector<particle_buffer1> pbuffer[n_proc];
    // Iterate through particles and send them to corresponding processors
    for(int i = nparticle_before[rank]; i < (nparticle_before[rank] + nparticle[rank]); i++) {
      int nx = floor(particles[i].x/bin_size);
      int ny = floor(particles[i].y/bin_size);
      if(nx == nbin_1d) nx--;
      if(ny == nbin_1d) ny--;
      int dest_temp;
      if(ny+1 <= residual_b*(nbinrow_pp+1)) dest_temp = floor(ny/(nbinrow_pp+1));
      else dest_temp = residual_b + floor((ny-residual_b*(nbinrow_pp+1))/nbinrow_pp);

      particle_buffer1 pbuffer_temp;
      pbuffer_temp.particle = particles[i];
      pbuffer_temp.bin_idx = nx;
      pbuffer_temp.bin_idy = ny;
      pbuffer[dest_temp].push_back(pbuffer_temp);
    }

    // Calculate the number of particles need to be sent and received for each processor
    int send_size[n_proc];
    for(int i = 0; i < n_proc; i++) {
      send_size[i] = pbuffer[i].size();
    }
    int rev_size_proc[n_proc]; 
    MPI_Alltoall(&send_size[0], 1, MPI_INT, &rev_size_proc[0], 1, MPI_INT, MPI_COMM_WORLD);
    int rev_buffer_size = 0;
    for(int i = 0; i < n_proc; i++) {
      rev_buffer_size += rev_size_proc[i];
    }

    // Calculate total number of particles to send, arrange them in an array
    int send_total = 0;   
    std::vector<particle_buffer1> rev_buffer(rev_buffer_size);
    int send_disp[n_proc];
    int rev_disp[n_proc];
    memset(send_disp, 0, n_proc*sizeof(int));
    memset(rev_disp, 0, n_proc*sizeof(int));
    for(int i = 0; i < n_proc; i++) {
      for(int j = 0; j < i; j++) {
        send_disp[i] += send_size[j];
        rev_disp[i] += rev_size_proc[j];
      }
    if(i == n_proc-1) send_total = send_disp[i] + send_size[i];
    }
  
    int count = 0;
    particle_buffer1 send_buffer[send_total];
    for(int i = 0; i < n_proc; i++) {
      for(int j = 0; j < pbuffer[i].size(); j++) {
        send_buffer[count] = pbuffer[i].at(j);
        count++;
      }
    }

    // Exchange particles between processors
    MPI_Alltoallv(&send_buffer[0], send_size, send_disp, PBUFFER1, rev_buffer.data(), rev_size_proc, rev_disp, PBUFFER1, MPI_COMM_WORLD);

    // Put received particles into bins
    for (int i = 0; i < rev_buffer_size; i++) {
      rev_buffer.at(i).particle.ax = rev_buffer.at(i).particle.ay = 0;
      bin[rev_buffer.at(i).bin_idx][rev_buffer.at(i).bin_idy].particle.push_back(rev_buffer.at(i).particle);
    }

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    MPI_Status status;
    for( int step = 0; step < NSTEPS; step++ )
    {
        navg = 0;
        dmin = 1.0;
        davg = 0.0;

        //
        //  save current step if necessary (slightly different semantics than in other codes)
        //
        if( find_option( argc, argv, "-no" ) == -1 )
          if( fsave && (step%SAVEFREQ) == 0 )
            save( fsave, n, particles );
        

	// Exchange information of particles near edge     
	std::vector<particle_t> send_upper;
	std::vector<particle_t> send_lower;
        int upper_size = 0;
        int lower_size = 0;
        int send_upper_size = 0;
        int send_lower_size = 0;

        // Calculate the number of particles at edge need to be sent and received
	if(rank != 0) {
	  for(int i = 0; i < nbin_1d; i++) {
	    bin_t temp_bin = bin[i][nbinrow_before[rank]];
	    for(int j = 0; j < temp_bin.particle.size(); j++) {
	      send_upper.push_back(temp_bin.particle.at(j));
	    }
	  }   
          send_upper_size = send_upper.size();
	  MPI_Send(&send_upper_size, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD);
	  MPI_Recv(&upper_size, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, &status);
	}

	if(rank != n_proc-1) {
	  for(int i = 0; i < nbin_1d; i++) {
	    bin_t temp_bin = bin[i][nbinrow_before[rank] + nbinrow[rank] - 1];
	    for(int j = 0; j < temp_bin.particle.size(); j++) {
	      send_lower.push_back(temp_bin.particle.at(j));
	    }
	  }
	  send_lower_size = send_lower.size();
	  MPI_Send(&send_lower_size, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
	  MPI_Recv(&lower_size, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD, &status);
	}

        // Send and receive particles with adjecant processors
	std::vector<particle_t> particle_upper(upper_size);
	std::vector<particle_t> particle_lower(lower_size);
      
        if(rank != 0) {
	  MPI_Send(&send_upper[0], send_upper_size, PARTICLE, rank-1, 0, MPI_COMM_WORLD);
	  MPI_Recv(&particle_upper[0], upper_size, PARTICLE, rank-1, 0, MPI_COMM_WORLD, &status);
        }
	if(rank != n_proc-1) {
	  MPI_Send(&send_lower[0], send_lower_size, PARTICLE, rank+1, 0, MPI_COMM_WORLD);
	  MPI_Recv(&particle_lower[0], lower_size, PARTICLE, rank+1, 0, MPI_COMM_WORLD, &status);
	}

        // Put the received particles into corresponding bins
	for(int i = 0; i < particle_upper.size(); i++) {
	  particle_t particle_temp = particle_upper.at(i);
	  int nx = floor(particle_temp.x/bin_size);
	  if(nx == nbin_1d) nx--; 
	  bin[nx][nbinrow_before[rank]-1].particle.push_back(particle_temp);
	}
	for(int i = 0; i < particle_lower.size(); i++) {
	  particle_t particle_temp = particle_lower.at(i);
	  int nx = floor(particle_temp.x/bin_size);
	  if(nx == nbin_1d) nx--;
	  bin[nx][nbinrow_before[rank] + nbinrow[rank]].particle.push_back(particle_temp);
	}

        //
        //  iterate through bins to compute forces
        //
	for(int row = nbinrow_before[rank]; row < nbinrow_before[rank] + nbinrow[rank]; row++) {
	  for(int col = 0; col < nbin_1d; col++) {
	    // Iterate particles inside this bin to calculate force
	    for(int i = 0; i < bin[col][row].particle.size(); i++) {
	      // Interaction with particles inside this bin
	      for(int j = 0; j < bin[col][row].particle.size(); j++) {
		if(j != i) {
		  apply_force(bin[col][row].particle.at(i), bin[col][row].particle.at(j), &dmin, &davg, &navg);
		}
	      }
	      // Interaction with particles at neighboring bins
              // Iterate through neighboring bins
	      for(int nei = 0; nei < bin[col][row].neighbor_bin_id.size(); nei++) {
		int nei_col = bin[col][row].neighbor_bin_id.at(nei).at(0);
		int nei_row = bin[col][row].neighbor_bin_id.at(nei).at(1);
                // Iterate through particles in that bin
		for(int k = 0; k < bin[nei_col][nei_row].particle.size(); k++) {
		  apply_force(bin[col][row].particle.at(i), bin[nei_col][nei_row].particle.at(k), &dmin, &davg, &navg);
		}
	      }		
	    }
	  }
	}


        if( find_option( argc, argv, "-no" ) == -1 )
        {
          
          MPI_Reduce(&davg,&rdavg,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&navg,&rnavg,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&dmin,&rdmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);

 
          if (rank == 0){
            //
            // Computing statistical data
            //
            if (rnavg) {
              absavg +=  rdavg/rnavg;
              nabsavg++;
            }
            if (rdmin < absmin) absmin = rdmin;
          }
        }

        //
        //  move particles and sent particles to new processor
        //
	std::vector<particle_buffer1> pbuffer[n_proc];
	for(int row = nbinrow_before[rank]; row < nbinrow_before[rank] + nbinrow[rank]; row++) {
	  for(int col = 0; col < nbin_1d; col++) {
	    for(int i = 0; i < bin[col][row].particle.size(); i++) {
	      particle_t particle_temp = bin[col][row].particle.at(i);
	      move(particle_temp);            
              // Calculate the next processor that this particle belongs to
	      int nx = floor(particle_temp.x/bin_size);
	      int ny = floor(particle_temp.y/bin_size);
	      if(nx == nbin_1d) nx--;
	      if(ny == nbin_1d) ny--;
	      int dest_temp;
              if(ny+1 <= residual_b*(nbinrow_pp+1)) dest_temp = floor(ny/(nbinrow_pp+1));
              else dest_temp = residual_b + floor((ny-residual_b*(nbinrow_pp+1))/nbinrow_pp);
	      particle_temp.ax = particle_temp.ay = 0;

              particle_buffer1 pbuffer_temp;
              pbuffer_temp.particle = particle_temp;
              pbuffer_temp.bin_idx = nx;
              pbuffer_temp.bin_idy = ny;
              pbuffer[dest_temp].push_back(pbuffer_temp);
	    }
	  }
	}

        // Reset particles in the bin to 0
        for(int i = nbinrow_before[rank]-1; i < (nbinrow_before[rank] + nbinrow[rank] + 1); i++) {
	  for(int j = 0; j < nbin_1d; j++) {
	    bin[j][i].particle.clear();
	  }
	}

        // Exchange particles between processors
        int send_size[n_proc];
        for(int i = 0; i < n_proc; i++) {
          send_size[i] = pbuffer[i].size();
        }
        int rev_size_proc[n_proc]; 
        MPI_Alltoall(&send_size[0], 1, MPI_INT, &rev_size_proc[0], 1, MPI_INT, MPI_COMM_WORLD);
        int rev_buffer_size = 0;
        for(int i = 0; i < n_proc; i++) {
          rev_buffer_size += rev_size_proc[i];
        }

        int send_total = 0;   
        std::vector<particle_buffer1> rev_buffer(rev_buffer_size);
        int send_disp[n_proc];
        int rev_disp[n_proc];
        memset(send_disp, 0, n_proc*sizeof(int));
        memset(rev_disp, 0, n_proc*sizeof(int));
        for(int i = 0; i < n_proc; i++) {
          for(int j = 0; j < i; j++) {
            send_disp[i] += send_size[j];
            rev_disp[i] += rev_size_proc[j];
          }
        if(i == n_proc-1) send_total = send_disp[i] + send_size[i];
        }
  
        int count = 0;
        particle_buffer1 send_buffer[send_total];
        for(int i = 0; i < n_proc; i++) {
          for(int j = 0; j < pbuffer[i].size(); j++) {
            send_buffer[count] = pbuffer[i].at(j);
            count++;
          }
        }

        MPI_Alltoallv(&send_buffer[0], send_size, send_disp, PBUFFER1, rev_buffer.data(), rev_size_proc, rev_disp, PBUFFER1, MPI_COMM_WORLD);

	// Receive particles and assign them to corresponding bins
        for (int i = 0; i < rev_buffer_size; i++) {
          bin[rev_buffer.at(i).bin_idx][rev_buffer.at(i).bin_idy].particle.push_back(rev_buffer.at(i).particle);
        }

	MPI_Barrier(MPI_COMM_WORLD);
    }


    simulation_time = read_timer( ) - simulation_time;
  
    if (rank == 0) {  
      printf( "n = %d, simulation time = %g seconds", n, simulation_time);

      if( find_option( argc, argv, "-no" ) == -1 )
      {
        if (nabsavg) absavg /= nabsavg;
      // 
      //  -The minimum distance absmin between 2 particles during the run of the simulation
      //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
      //  -A simulation where particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
      //
      //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
      //
      printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
      if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
      if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
      }
      printf("\n");     
        
      //  
      // Printing summary data
      //  
      if( fsum)
        fprintf(fsum,"%d %d %g\n",n,n_proc,simulation_time);
    }
  
    //
    //  release resources
    //
    if ( fsum )
        fclose( fsum );
    //free( partition_offsets );
    //free( partition_sizes );
    //free( local );
    free( particles );
    if( fsave )
        fclose( fsave );
    
    MPI_Finalize( );
    
    return 0;
}
