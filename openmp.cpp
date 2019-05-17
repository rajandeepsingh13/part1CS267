#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include <array>
#include <vector>

#include "common.h"
#include "omp.h"

typedef struct
{
  std::vector<std::array<int, 2>> id;
  std::vector<int> particle_id;
  std::vector<std::array<int, 2>> neighbor_bin_id;
  //int nparticle;
} bin_t;

//
//  benchmarking program
//
int main( int argc, char **argv )
{   
    int navg,nabsavg=0,numthreads; 
    double dmin, absmin=1.0,davg,absavg=0.0;
	
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" ); 
        printf( "-no turns off all correctness checks and particle output\n");   
        return 0;
    }

    int n = read_int( argc, argv, "-n", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );

    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname ? fopen ( sumname, "a" ) : NULL;      

    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    set_size( n );
    init_particles( n, particles );

    // Initialize bin and assign neighboring bins
    double bin_size = cutoff;
    double area_size = sqrt( density * n );
    // Calculate # of bins in each direction
    int nbin_1d = floor(area_size/bin_size);
    if(fmod(area_size, bin_size) != 0) nbin_1d++;
    // Initialize 2D array of the bin, the 3rd dimension is just padding to prevent false sharing
    bin_t bin[nbin_1d][nbin_1d][8];

    // Assign neighboring bin id to each bin
    for(int row = 0; row < nbin_1d; row++) {
      for(int col = 0; col < nbin_1d; col++) {
        std::array<int, 2> id_temp = {row, col};
        bin[row][col][0].id.push_back(id_temp);
        if(row == 0) {
	  if(col == 0) {
            std::array<int, 2> temp[3];
            temp[0] = {0,1};
            temp[1] = {1,0};
            temp[2] = {1,1};
            for(int i = 0; i < sizeof(temp)/sizeof(*temp); i++)
	      bin[row][col][0].neighbor_bin_id.push_back(temp[i]);
	  } else if (col == nbin_1d-1) {
	    std::array<int, 2> temp[3];
            temp[0] = {row+1,col};
            temp[1] = {row,col-1};
            temp[2] = {row+1,col-1};
            for(int i = 0; i < sizeof(temp)/sizeof(*temp); i++)
	      bin[row][col][0].neighbor_bin_id.push_back(temp[i]);
    	  } else {
	    std::array<int, 2> temp[5];
            temp[0] = {row,col-1};
            temp[1] = {row,col+1};
            temp[2] = {row+1,col-1};
            temp[3] = {row+1,col};
            temp[4] = {row+1,col+1};
            for(int i = 0; i < sizeof(temp)/sizeof(*temp); i++)
	      bin[row][col][0].neighbor_bin_id.push_back(temp[i]);
    	  } 
	}    
        else if(row == nbin_1d-1) {
	  if(col == 0) {
            std::array<int, 2> temp[3];
            temp[0] = {row-1,0};
            temp[1] = {row-1,1};
            temp[2] = {row,1};
            for(int i = 0; i < sizeof(temp)/sizeof(*temp); i++)
	      bin[row][col][0].neighbor_bin_id.push_back(temp[i]);
	  } else if (col == nbin_1d-1) {
	    std::array<int, 2> temp[3];
            temp[0] = {row-1,col};
            temp[1] = {row-1,col-1};
            temp[2] = {row,col-1};
            for(int i = 0; i < sizeof(temp)/sizeof(*temp); i++)
	      bin[row][col][0].neighbor_bin_id.push_back(temp[i]);
    	  } else {
	    std::array<int, 2> temp[5];
            temp[0] = {row,col-1};
            temp[1] = {row,col+1};
            temp[2] = {row-1,col-1};
            temp[3] = {row-1,col};
            temp[4] = {row-1,col+1};
            for(int i = 0; i < sizeof(temp)/sizeof(*temp); i++)
	      bin[row][col][0].neighbor_bin_id.push_back(temp[i]);
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
	      bin[row][col][0].neighbor_bin_id.push_back(temp[i]);
	  } else if (col == nbin_1d-1) {
	    std::array<int, 2> temp[5];
            temp[0] = {row-1,col};
            temp[1] = {row-1,col-1};
            temp[2] = {row,col-1};
            temp[3] = {row+1,col-1};
            temp[4] = {row+1,col};
            for(int i = 0; i < sizeof(temp)/sizeof(*temp); i++)
	      bin[row][col][0].neighbor_bin_id.push_back(temp[i]);
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
	      bin[row][col][0].neighbor_bin_id.push_back(temp[i]);
    	  }
	}
      }
    }

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );

    // Initialize a 2D array of locks, one lock for each bin, so that can update different bin at the same time
    omp_lock_t writelock[nbin_1d][nbin_1d];
    for(int i = 0; i < nbin_1d; i++) {
      for(int j = 0; j < nbin_1d; j++) {
	omp_init_lock(&(writelock[i][j]));
      }
    }

    #pragma omp parallel private(dmin) 
    {
    numthreads = omp_get_num_threads();
    for( int step = 0; step < NSTEPS; step++ )
    {
        navg = 0;
        davg = 0.0;
	dmin = 1.0;

        // Reset particles in the bin to 0
	#pragma omp for
        for(int i = 0; i < nbin_1d; i++) {
	  for(int j = 0; j < nbin_1d; j++) {
	    bin[i][j][0].particle_id.clear();
	  }
	}

        // Assign particles to bin and set acceleration to 0
	#pragma omp for
        for(int i = 0; i < n; i++) {
	  int nx = floor(particles[i].x/bin_size);
	  int ny = floor(particles[i].y/bin_size);
	  if(nx == nbin_1d) nx--;
	  if(ny == nbin_1d) ny--;
          // Use lock to prevent data race
	  omp_set_lock(&(writelock[nx][ny]));
	  bin[nx][ny][0].particle_id.push_back(i);
	  omp_unset_lock(&(writelock[nx][ny]));
	  particles[i].ax = particles[i].ay = 0;
	}

        //
        //  iterate through bins to compute forces
        //
        #pragma omp for reduction (+:navg) reduction(+:davg)
	for(int row = 0; row < nbin_1d; row++) {
	  for(int col = 0; col < nbin_1d; col++) {
	    bin_t temp_bin = bin[row][col][0];
	    // Iterate particles inside this bin to calculate force
	    for(int i = 0; i < temp_bin.particle_id.size(); i++) {
	      int id_this = temp_bin.particle_id.at(i);
	      // Interaction with particles inside this bin
	      for(int j = 0; j < temp_bin.particle_id.size(); j++) {
		if(j != i) {
		  int id_neip = temp_bin.particle_id.at(j);
		  apply_force(particles[id_this], particles[id_neip], &dmin, &davg, &navg);
		}
	      }
	      // Interaction with particles at neighboring bins
              // Iterate through neighboring bins
	      for(int nei = 0; nei < temp_bin.neighbor_bin_id.size(); nei++) {
		int nei_row = temp_bin.neighbor_bin_id.at(nei).at(0);
		int nei_col = temp_bin.neighbor_bin_id.at(nei).at(1);
		bin_t temp_nei = bin[nei_row][nei_col][0];
                // Iterate through particles in that bin
		for(int i = 0; i < temp_nei.particle_id.size(); i++) {
		  int id_neip = temp_nei.particle_id.at(i);
		  apply_force(particles[id_this], particles[id_neip], &dmin, &davg, &navg);
		}
	      }		
	    }
	  }
	}

        //
        //  move particles
        //
        #pragma omp for
        for( int i = 0; i < n; i++ ) 
            move( particles[i] );
  
        if( find_option( argc, argv, "-no" ) == -1 ) 
        {
          //
          //  compute statistical data
          //
          #pragma omp master
          if (navg) { 
            absavg += davg/navg;
            nabsavg++;
          }

          #pragma omp critical
	  if (dmin < absmin) absmin = dmin; 
		
          //
          //  save if necessary
          //
          #pragma omp master
          if( fsave && (step%SAVEFREQ) == 0 )
              save( fsave, n, particles );
        }
    }
}

    simulation_time = read_timer( ) - simulation_time;
    
    printf( "n = %d,threads = %d, simulation time = %g seconds", n,numthreads, simulation_time);

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
        fprintf(fsum,"%d %d %g\n",n,numthreads,simulation_time);

    //
    // Clearing space
    //
    if( fsum )
        fclose( fsum );

    free( particles );
    if( fsave )
        fclose( fsave );
    
    return 0;
}
