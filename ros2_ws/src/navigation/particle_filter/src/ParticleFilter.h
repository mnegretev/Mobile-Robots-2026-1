/*
 * MOBILE ROBOTS - UNAM, FI, 2026-1
 * LOCALIZATION BY PARTICLE FILTERS
 *
 * Instructions:
 * Write the code necessary to implement localization by particle filters.
 * Modify only the sections marked with the TODO comment. 
 */
#include <cmath> 
#include <vector>
#include <random>
#include "particle_filter/ray_tracer.h"

class ParticleFilter
{
public:
    ParticleFilter(){}

    static std::vector<geometry_msgs::msg::Pose2D> get_initial_distribution(
    int N, float min_x, float max_x, float min_y, float max_y, float min_a, float max_a)
    {
	random_numbers::RandomNumberGenerator rnd;
	std::vector<geometry_msgs::msg::Pose2D> particles(N);
	for(int i = 0; i < N; i++) {
            particles[i].x = rnd.uniformReal(min_x, max_x);
            particles[i].y = rnd.uniformReal(min_y, max_y);
            particles[i].theta = rnd.uniformReal(min_a, max_a);
        }
        return particles;
    }

    static void move_particles(std::vector<geometry_msgs::msg::Pose2D>& particles,
			       float delta_x, float delta_y, float delta_t, float sigma2)
    {
	random_numbers::RandomNumberGenerator rnd;
	for (size_t i = 0; i < particles.size(); ++i) {
	    float theta_i = particles[i].theta;
	    float noise_x = rnd.gaussian(0.0, sigma2);
	    float noise_y = rnd.gaussian(0.0, sigma2);
	    float noise_theta = rnd.gaussian(0.0, sigma2);
	    particles[i].x += delta_x * cos(theta_i) - delta_y * sin(theta_i) + noise_x;
	    particles[i].y += delta_x * sin(theta_i) + delta_y * cos(theta_i) + noise_y;
	    particles[i].theta += delta_t + noise_theta;
	}
    }

    static std::vector<sensor_msgs::msg::LaserScan> simulate_particle_scans(
	std::vector<geometry_msgs::msg::Pose2D>& particles,
	nav_msgs::msg::OccupancyGrid& map,
	sensor_msgs::msg::LaserScan& sensor_specs)
    {
	/*
	 * TODO:
	 * Review the code to simulate a laser scan for each particle given the set of particles and a static map. 
	 */
	std::vector<sensor_msgs::msg::LaserScan> simulated_scans(particles.size());
	for(size_t i=0; i < particles.size(); i++)
	{
	    geometry_msgs::msg::Pose sensor_pose;
	    sensor_pose.position.x    = particles[i].x;
	    sensor_pose.position.y    = particles[i].y;
	    sensor_pose.orientation.w = cos(particles[i].theta/2);
	    sensor_pose.orientation.z = sin(particles[i].theta/2);
	    
	    simulated_scans[i] = ray_tracer::simulateRangeScan(map, sensor_pose, sensor_specs);
	    
	}
	return simulated_scans;
    }

    static std::vector<double> get_particle_similarities(
	std::vector<sensor_msgs::msg::LaserScan>& simulated_scans,
	sensor_msgs::msg::LaserScan& real_scan,
	int downsampling, float sigma2)
    {
	std::vector<double> similarities;
	similarities.resize(simulated_scans.size());
      	
	float max_range = real_scan.range_max;  // Rango máximo para penalizaciones en lecturas inválidas.
	double total_delta_sum = 0.0;  // Para calcular pesos antes de normalizar.
	int num_ranges = simulated_scans[0].ranges.size();  // Asumiendo todas las simuladas tienen el mismo tamaño.
	for (size_t i = 0; i < simulated_scans.size(); ++i) {
	    double delta = 0.0;
	    for (size_t j = 0; j < num_ranges; ++j) {
		float sim_range = simulated_scans[i].ranges[j];
		size_t real_idx = j * downsampling;
		if (real_idx < real_scan.ranges.size()) {
		    float real_range = real_scan.ranges[real_idx];
		    if (std::isfinite(sim_range) && std::isfinite(real_range) && sim_range < max_range && real_range < max_range) {
			delta += std::fabs(sim_range - real_range);  // Diferencia absoluta solo si ambos válidos.
		    } else {
			delta += max_range;  // Penalización por lecturas inválidas (inf o out-of-range).
		    }
		} else {
		    delta += max_range;  // Si downsampling excede el tamaño del real, penalizar.
		}
	    }
	    delta /= num_ranges;  // Error medio absoluto (MAE).
	    similarities[i] = std::exp(-delta / sigma2);  // Similitud gaussiana como peso.
	    total_delta_sum += similarities[i];
	}
	// Normalización: pesos suman 1.0 para distribución de probabilidad.
	if (total_delta_sum > 0.0) {
	    for (size_t i = 0; i < similarities.size(); ++i) {
		similarities[i] /= total_delta_sum;
	    }
	} else {
	    // Caso degenerado: asignar uniformemente si todas similitudes son 0.
	    for (size_t i = 0; i < similarities.size(); ++i) {
		similarities[i] = 1.0 / similarities.size();
	    }
	}
	
	return similarities;
    }
    
    static int random_choice(std::vector<double>& probabilities)
    {
	random_numbers::RandomNumberGenerator rnd;
	/*
	 * TODO:
	 *
	 * Write an algorithm to choice an integer in the range [0, N-1], with N, the length of 'probabilities'.
	 * Probability of picking an integer 'i' is given by the corresponding probabilities[i] value.
	 * Return the chosen integer.
	 * x = rnd.uniformReal(0,1)
	 * FOR i=[0...probabilities.size())
	 *    IF x < probabilities[i]
	 *        return i
	 *    ELSE
	 *        x -= probabilities[i]
	 * return -1
	 */
	double x = rnd.uniformReal(0.0, 1.0);
	for (size_t i = 0; i < probabilities.size(); ++i) {
	    if (x < probabilities[i]) {
		return static_cast<int>(i);  // Retorna el índice seleccionado.
	    }
	    x -= probabilities[i];  // Acumula: método de rueda para distribución discreta.
	}
	
	return -1;
    }

    static std::vector<geometry_msgs::msg::Pose2D> resample_particles(
	std::vector<geometry_msgs::msg::Pose2D>& particles, std::vector<double>& probabilities, float sigma2)
    {

	random_numbers::RandomNumberGenerator rnd;
	std::vector<geometry_msgs::msg::Pose2D> resampled_particles(particles.size());
	/*
	 * TODO:
	 * Sample, with replacement, N particles from the set 'particles'.
	 * The probability of the i-th particle to be resampled is given by probabilities[i].
	 * Use the random_choice function to pick a particle with the correct probability.
	 * Add gaussian noise to each sampled particle (add noise to x,y and theta). Use sigma2 as noise variance.
	 */
	
	for (size_t j = 0; j < particles.size(); ++j) {
	    int idx = random_choice(probabilities);  // Selecciona índice con P(i) = probabilities[i].
	    if (idx >= 0) {  // Verifica selección válida.
		resampled_particles[j] = particles[idx];  // Copia la partícula seleccionada.
		// Agrega ruido gaussiano a x, y, theta para diversificar.
		resampled_particles[j].x += rnd.gaussian(0.0, sigma2);
		resampled_particles[j].y += rnd.gaussian(0.0, sigma2);
		resampled_particles[j].theta += rnd.gaussian(0.0, sigma2);
	    } else {
		// Fallback raro: copia una partícula aleatoria si random_choice falla.
		resampled_particles[j] = particles[j % particles.size()];
		resampled_particles[j].x += rnd.gaussian(0.0, sigma2);
		resampled_particles[j].y += rnd.gaussian(0.0, sigma2);
		resampled_particles[j].theta += rnd.gaussian(0.0, sigma2);
	    }
	}
	
	
	return resampled_particles;
    }
    
};
