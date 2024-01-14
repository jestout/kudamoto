#include <iostream>
#include <fstream>
#include <cmath>
#include <complex>
#include <string>
#include <sstream>
#include <vector>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/cauchy_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/lexical_cast.hpp>

const double Pi = 3.1415926535;


//#define RK4


//#define DEBUG

// Random Number Generators and Distributions and ...
boost::mt19937 gen;

boost::cauchy_distribution<> dist_cauchy(0, 0.2);
boost::variate_generator<boost::mt19937&, boost::cauchy_distribution<> > random_omega(gen,dist_cauchy);

boost::uniform_real<> dist_Pi(0, 2*Pi);
boost::variate_generator<boost::mt19937&, boost::uniform_real<> > random_theta(gen, dist_Pi);

boost::uniform_real<> dist_bool(0, 1);
boost::variate_generator<boost::mt19937&, boost::uniform_real<> > rando(gen, dist_bool);

boost::uniform_real<> dist_neg(-1, 1);
boost::variate_generator<boost::mt19937&, boost::uniform_real<> > neg_rando(gen, dist_neg);


// Function Prototypes

std::complex<double> calc_order_param(boost::numeric::ublas::vector<double> & theta);
double ** gen_barabasi(int initial_node_count, int final_node_count, double probability);
double ** gen_config_model(int N, double avg_degree, double power);
double ** gen_erdos(int N, double probability);
double largest_eigenvalue(double ** a, int N);
double ** calc_threshold_matrix(double ** a, std::vector<boost::numeric::ublas::vector<double> > & theta_list);
double calc_r_link(double ** a, double ** D, int N);
int * degree_dist(double ** a, int N);
double ** calc_rho(std::vector<boost::numeric::ublas::vector<double> > & theta_list);


void vector_init(boost::numeric::ublas::vector<double> & vect, boost::variate_generator<boost::mt19937&, boost::uniform_real<> >& random_num)
{
	int N = vect.size();

	for(int i = 0; i < N; i++)
		vect[i] = random_num();

}

void vector_init(boost::numeric::ublas::vector<double> & vect, boost::variate_generator<boost::mt19937&, boost::cauchy_distribution<> >& random_num)
{
	int N = vect.size();

	for(int i = 0; i < N; i++)
		vect[i] = random_num();
}

void k_f(boost::numeric::ublas::vector<double> & output, boost::numeric::ublas::vector<double> & theta, boost::numeric::ublas::vector<double> & omega, double ** a, double K)
{
	// Should all be the same size
	int N = theta.size();
	std::complex<double> temp;
	
	for(int i = 0; i < N; i++)
	{
		temp = 0;

		// Sum over j
		for(int j = 0; j < N; j++)
		{
			temp += a[i][j]*std::polar(1.0, theta[j]);
		}

		output[i] = omega[i] + (K/N)*std::imag(std::polar(1.0, -1.0*theta[i])*temp);
	}
}



int main(int argc, char ** argv)
{

	int N;							// Number of Oscillators
	int seed;						// Seed for mt19937
	double K;						// Coupling Constant
	double step_size;					// Step Size for RK4
	double end_time;					// Total Time to Run

	std::string matrix_creation_method;			// Need for string comparisons
	std::string file_output_string;				// Need for convenience

	double ** a;						// Adjacency Matrix

	double probability;					// Probability for barabasi_albert and erdos_renyi
	int init_node_count;					// init_node_count for barabasi_albert
	int avg_degree;						// average degree for config model
	double power;						// power for config model
	
	std::fstream ofile;					// Output File
	std::ostringstream temp_sstream;			// Temp String Stream for some conversions

	double lambda;
	
	// We can have several types of arguments
	// ./cont_kuro_fast [Number of Oscillators] [Coupling Constant] [Stepsize] [End Time] [Seed] --erdos_renyi [Probability of Connection] [Output]
	// ./cont_kuro_fast [Number of Oscillators] [Coupling Constant] [Stepsize] [End Time] [Seed] --barabasi_albert [Initial Node Count] [Probability of Connection] [Output]
	// ./cont_kuro_fast [Number of Oscillators] [Coupling Constant] [Stepsize] [End Time] [Seed] --config_model [Average Degree] [Power] [Output]

	if(argc == 9 || argc == 10)
	{

		// These arguments are easy, always in the same place
		N = boost::lexical_cast<int>(argv[1]);
		K = boost::lexical_cast<double>(argv[2]);
		step_size = boost::lexical_cast<double>(argv[3]);
		end_time = boost::lexical_cast<double>(argv[4]);
		seed = boost::lexical_cast<int>(argv[5]);
		matrix_creation_method = argv[6];

		std::cout << "Number of Oscillators: " << N << std::endl << "Coupling Constant: " << K << std::endl << "End Time: " << end_time << " s" << std::endl << "Stepsize: " << step_size << " s" << std::endl << "Seed: " << seed << std::endl;

		gen.seed(seed);
		
		if(matrix_creation_method == "--erdos_renyi")
		{
			probability = boost::lexical_cast<double>(argv[7]);
			a = gen_erdos(N, probability);

			ofile.open(argv[8], std::fstream::out);
			
			file_output_string = "Matrix Generation Method: Erdos Renyi\nProbability: ";
			temp_sstream << probability;
			file_output_string.append(temp_sstream.str());
			std::cout << file_output_string << std::endl;
			
		} else if(matrix_creation_method == "--barabasi_albert")
		{
			init_node_count = boost::lexical_cast<int>(argv[7]);
			probability = boost::lexical_cast<double>(argv[8]);

			a = gen_barabasi(init_node_count, N, probability);

			ofile.open(argv[9], std::fstream::out);

			file_output_string = "Matrix Generation Method: Barabasi-Albert\nInitial Node Count: ";
			temp_sstream << init_node_count;
			file_output_string.append(temp_sstream.str());
			file_output_string.append("\nProbability: ");
			temp_sstream << probability;
			file_output_string.append(temp_sstream.str());
			std::cout << file_output_string << std::endl;
			
		} else if(matrix_creation_method == "--config_model")
		{
			avg_degree = boost::lexical_cast<int>(argv[7]);
			power = boost::lexical_cast<double>(argv[8]);

			a = gen_config_model(N, avg_degree, power);

			ofile.open(argv[9], std::fstream::out);

			file_output_string = "Matrix Generation Method: Configuration Model\nAverage Degree: ";
			temp_sstream << avg_degree;
			file_output_string.append(temp_sstream.str());
			file_output_string.append("\nPower: ");
			temp_sstream << power;
			file_output_string.append(temp_sstream.str());
			std::cout << file_output_string << std::endl;
		} else
		{
			
			std::cout << "Arguments must be in either of the following forms:" << std::endl;
			std::cout << "./cont_kura_fast [Number of Oscillators] [Coupling Constant] [Stepsize] [End Time] [Seed] --erdos_renyi [Probability of Connection] [Output]" << std::endl;
			std::cout << "./cont_kura_fast [Number of Oscillators] [Coupling Constant] [Stepsize] [End Time] [Seed] --barabasi_albert [Initial Node Count] [Probability of Connection] [Output]" << std::endl;
			std::cout << "./cont_kura_fast [Number of Oscillators] [Coupling Constant] [Stepsize] [End Time] [Seed] --config_model [Average Degree] [Power] [Output]" << std::endl;
			return 1;
		}


		// Check if we were successful in opening the output file, if not shut down.
		if(!ofile.is_open())
		{
			std::cout << std::endl << "Unable to open output file." << std::endl;
			return 1;
		}

		// Print out to file
		ofile << "Number of Oscillators: " << N << std::endl << "Coupling Constant: " << K << std::endl << "End Time: " << end_time << " s" << std::endl << "Stepsize: " << step_size << " s" << std::endl << "Seed: " << seed << std::endl << file_output_string << std::endl;
		
	} else
	{
		std::cout << "Arguments must be in either of the following forms:" << std::endl;
		std::cout << "./cont_kura_fast [Number of Oscillators] [Coupling Constant] [Stepsize] [End Time] [Seed] --erdos_renyi [Probability of Connection] [Output]" << std::endl;
		std::cout << "./cont_kura_fast [Number of Oscillators] [Coupling Constant] [Stepsize] [End Time] [Seed] --barabasi_albert [Initial Node Count] [Probability of Connection] [Output]" << std::endl;
		std::cout << "./cont_kura_fast [Number of Oscillators] [Coupling Constant] [Stepsize] [End Time] [Seed] --config_model [Average Degree] [Power] [Output]" << std::endl;
		return 1;
	}

	// Now create and initialize the omega array, theta array, mt19937 generator, and distributions


	boost::numeric::ublas::vector<double> omega(N);
	boost::numeric::ublas::vector<double> theta(N);
	boost::numeric::ublas::vector<double> d_theta(N);

	vector_init(omega, random_omega);
	vector_init(theta, random_theta);
	
	//init_rand_vec(theta, random_theta);
	//init_rand_vec(omega, random_theta);
	
	// Now print Theta values
	ofile << std::endl << "Theta Values: " << std::endl;

	for(int i = 0; i < N; i++)
		ofile << i + 1 << ": " << theta[i] << std::endl;

	ofile << std::endl << "Omega Values: " << std::endl;

	for(int i = 0; i < N; i++)
		ofile << i + 1 << ": " << omega[i] << std::endl;

	// Now lets print the adjacency matrix we're using
	ofile << std::endl << "Adjacency Matrix: " << std::endl;
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
		{
			ofile << a[i][j] << " ";
		}

		ofile << std::endl;
	}
	

	lambda = largest_eigenvalue(a, N);
	
	// Now lets integrate forward

	boost::numeric::ublas::vector<double> k1(N);
	boost::numeric::ublas::vector<double> k2(N);
	boost::numeric::ublas::vector<double> k3(N);
	boost::numeric::ublas::vector<double> k4(N);
	boost::numeric::ublas::vector<double> temp(N);
	boost::numeric::ublas::vector<double> average_theta(N);
	boost::numeric::ublas::vector<double> average_d_theta(N);

	std::vector<boost::numeric::ublas::vector<double> > theta_hist;
	std::vector<boost::numeric::ublas::vector<double> > d_theta_hist;
	
	//void k_f(boost::numeric::ublas::vector<double> & output, boost::numeric::ublas::vector<double> & theta, boost::numeric::ublas::vector<double> & omega, double ** a, double K, double factor)

	for(double t = 0.0; t <= end_time; t += step_size)
	{
		//ofile << "Time " << t << ": " << std::abs(calc_order_param(theta)) << std::endl;
		
		
		//std::cout << "Time : " << t << std::endl;	
			
		
		// Calculate k1 = 0.5*h*f(theta)
		k_f(k1, theta, omega, a, K);

		// Calculate k2 = 0.5*h*f(theta + k1)
		temp = theta + 0.5*step_size*k1;
		k_f(k2, temp, omega, a, K);
		
		temp = theta + 0.5*step_size*k2;
		k_f(k3, temp, omega, a, K);

		temp = theta + step_size*k3;
		k_f(k4, temp, omega, a, K);

		if(t >= 0.95*end_time)
		{	
			theta_hist.push_back(theta);
			d_theta_hist.push_back(k1);
		}

		theta = theta + (k1 + 2*k2 + 2*k3 + k4)/6;
		
		
	}
	
	// Now process theta_hist and d_theta_hist

	std::vector<boost::numeric::ublas::vector<double> >::iterator theta_it;
	std::vector<boost::numeric::ublas::vector<double> >::iterator d_theta_it;
	std::complex<double> temp_order = 0;
	double ** D;
	double ** rho;
	int * deg_dist;
	
	average_theta.clear();
	average_d_theta.clear();

	for(theta_it = theta_hist.begin(), d_theta_it = d_theta_hist.begin(); (theta_it < theta_hist.end()) && (d_theta_it < d_theta_hist.end()); theta_it++, d_theta_it++)
	{
		average_theta += (*theta_it);
		average_d_theta += *d_theta_it;

		// Also Calculate Order Parameter
		temp_order += calc_order_param(*theta_it);
	}

	

	
	average_theta /= theta_hist.size();
	average_d_theta /= d_theta_hist.size();
	temp_order /= theta_hist.size();

	ofile << "Average Theta" << std::endl << average_theta << std::endl << "Average Theta Dot" << std::endl << average_d_theta << std::endl;
	std::cout << "Regular Order Parameter: " << std::abs(temp_order) << std::endl;
	ofile << "Regular Order Parameter: " << std::abs(temp_order) << std::endl;

	D = calc_threshold_matrix(a, theta_hist);
	rho = calc_rho(theta_hist);
	deg_dist = degree_dist(a, N);
	
	double c, d;

	// Print out D
	ofile << std::endl << "D: " << std::endl;
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
		{
			c = fabs(omega[i] - omega[j]);
			d = deg_dist[i] + deg_dist[j];
			
			if(i != j)	
				ofile << D[i][j] << " " << c << " " << d << " " << d/c << " " << c*d << std::endl;
		}

	}

	// Calculate r_link
	ofile << std::endl << "r_link: " << calc_r_link(a, D, N) << std::endl;
	std::cout << "r_link: " << calc_r_link(a, D, N) << std::endl;

	// Clean up

	for(int i = 0; i < N; i++)
	{
		delete a[i];
		delete D[i];
		delete rho[i];
	}
	
	delete D;
	delete rho;
	delete a;
	delete deg_dist;
	
	

	ofile.close();
	

	return 0;
}

double ** gen_config_model(int N, double avg_degree, double power)
{
	double normalization_const = 0.0;
	
	// Create degree distribution
	double * v = new double[N];
	
	// Create adjacency matrix
	double ** a = new double*[N];

	for(int i = 0; i < N; i++)
		a[i] = new double[N];

	for(int i = 0; i < N; i++)
	{
		v[i] = pow(i+1, -1/(power-1));

		#ifdef DEBUG
		std::cout << "v[" << i << "]: " << v[i] << std::endl;
		#endif
	}

	for(int i = 0; i < N; i++)
		for(int j = 0; j < i; j++)
			normalization_const += v[i]*v[j];

	normalization_const = 0.5*avg_degree*N/normalization_const;
	
	for(int i = 0; i < N; i++)
	{	
		for(int j = 0; j < i; j++)
		{
			a[i][j] = ((normalization_const*v[i]*v[j])>rando())?1:0;
			a[j][i] = a[i][j];
		}

		a[i][i] = 0;
	}

	delete v;
	
	return a;
}


double ** gen_erdos(int N, double probability)
{
	double ** a = new double*[N];

	for(int i = 0; i < N; i++)
		a[i] = new double[N];

	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < i; j++)
		{
			if(rando() < probability)
			{
				a[i][j] = 1;
				a[j][i] = 1;
			} else
			{
				a[i][j] = 0;
				a[j][i] = 0;
			}
		}
		a[i][i] = 0;
	}

	#ifdef DEBUG
	std::cout << std::endl << "Created a: " << std::endl;
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
		{
			std::cout << a[i][j] << " ";
		}

		std::cout << std::endl;
	}
	#endif
		

	return a;
}

double ** gen_barabasi(int initial_node_count, int final_node_count, double probability)
{
	double ** a = new double*[final_node_count];
	int * v = new int[final_node_count];
	int sum_k;

	for(int i = 0; i < final_node_count; i++)
		a[i] = new double[final_node_count];
		
	for(int i = 0; i < initial_node_count; i++)
		v[i] = 0;

	for(int i = 0; i < initial_node_count; i++)
	{
		
		for(int j = 0; j < i; j++)
		{
			if(rando() < probability)
			{
				a[i][j] = 1;
				a[j][i] = 1;
				v[i]++;
				v[j]++;
			} else
			{
				a[i][j] = 0;
				a[j][i] = 0;
			}
		}
		a[i][i] = 0;
	}

	std::cout << std::endl << "Created Erdos Array: " << std::endl;

	for(int i = 0; i < final_node_count; i++)
	{
		for(int j = 0; j < final_node_count; j++)
			std::cout << a[i][j] << " ";
		std::cout << std::endl;
	} 
	
	// Now start attaching nodes
	for(int i = initial_node_count; i < final_node_count; i++)
	{
		sum_k = 0;

		// Find Degree Distribution sum
		for(int j = 0; j < i; j++)
		{
			for(int k = 0; k < j; k++)
				sum_k += static_cast<int>(a[j][k]);
		}

		for(int j = 0; j < i; j++)
		{
			if(rando() < (static_cast<double>(v[j])/static_cast<double>(sum_k)))
			{
				a[i][j] = 1;
				a[j][i] = 1;
				v[i]++;
				v[j]++;
			} else
			{
				a[i][j] = 0;
				a[j][i] = 0;
			}
		}
	}
		

	delete v;
	
	return a;

}

std::complex<double> calc_order_param(boost::numeric::ublas::vector<double>& theta)
{
	std::complex<double> result = 0;
	int N = theta.size();
	
	for(int i = 0; i < N; i++)
	{
		result += std::polar(1.0, theta[i]);
	}

	return result/(static_cast<double>(N));
}

double largest_eigenvalue(double ** a, int N)
{
	double * v = new double[N];
	double * v_prime = new double[N];

	double mag_result = 0;
	double dot_result;
	
	for(int i = 0; i < N; i++)
	{	
		v[i] = neg_rando();
		mag_result += v[i]*v[i];
	}

	mag_result = sqrt(mag_result);

	for(int i = 0; i < N; i++)
		v[i] /= mag_result;
	

	do
	{
		mag_result = 0;
		dot_result = 0;

		for(int i = 0; i < N; i++)
		{
			v_prime[i] = 0;

			for(int j = 0; j < N; j++)
				v_prime[i] += a[j][i]*v[j];

			dot_result += v_prime[i]*v[i];
			mag_result += v_prime[i]*v_prime[i];
		}

		mag_result = sqrt(mag_result);
		dot_result /= mag_result;

		if(v_prime[0] < 0)
			mag_result = -mag_result;

		for(int i = 0; i < N; i++)
			v[i] = v_prime[i]/mag_result;
	} while((1.0 - dot_result) > 1.0e-14);

	return  mag_result;
}

double ** calc_threshold_matrix(double ** a, std::vector<boost::numeric::ublas::vector<double> > & theta_list)
{
	std::vector<boost::numeric::ublas::vector<double> >::iterator theta_it = theta_list.begin();
	int N = (*theta_it).size();
	std::complex<double> temp = 0;

	// Now create D matrix
	double ** D = new double*[N];

	for(int i = 0; i < N; i++)
		D[i] = new double[N];

	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
		{
			temp = 0;
			
			// for each element in D, calculate the magnitude of this integral
			for(theta_it = theta_list.begin(); theta_it < theta_list.end(); theta_it++)
			{
				temp += std::polar(1.0, (*theta_it)[i] - (*theta_it)[j]);
			}

			temp /= theta_list.size();

			D[i][j] = a[i][j]*std::abs(temp);
		}
	}

	return D;
}

double ** calc_rho(std::vector<boost::numeric::ublas::vector<double> > & theta_list)
{
	std::vector<boost::numeric::ublas::vector<double> >::iterator theta_it = theta_list.begin();
	int N = (*theta_it).size();
	double temp;
	
	double ** rho = new double*[N];
	
	for(int i = 0; i < N; i++)
		rho[i] = new double[N];
	
	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
		{
			temp = 0;
			
			for(theta_it = theta_list.begin(); theta_it < theta_list.end(); theta_it++)
			{
				temp += cos((*theta_it)[i] - (*theta_it)[j]);
			}
			
			temp /= theta_list.size();
			
			rho[i][j] = temp;
		}
	}
	
	return rho;
}
			
double calc_r_link(double ** a, double ** D, int N)
{
	double temp = 0.0;
	double two_n = 0.0;

	for(int i = 0; i < N; i++)
	{
		for(int j = 0; j < N; j++)
		{
			temp += D[i][j];
			two_n += a[i][j];
		}
	}

	return temp/two_n;
}

int * degree_dist(double ** a, int N)
{
	int * dist = new int[N];
	
	for(int i = 0; i < N; i++)
	{
		dist[i] = 0;
		
		for(int j = 0; j < N; j++)
		{
			dist[i] += static_cast<int>(a[i][j]);
		}
	}
	
	return dist;
}
