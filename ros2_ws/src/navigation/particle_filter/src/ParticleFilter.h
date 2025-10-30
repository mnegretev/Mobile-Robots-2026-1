/*
 * MOBILE ROBOTS - UNAM, FI, 2026-1
 * LOCALIZATION BY PARTICLE FILTERS
 *
 * Instructions:
 * Write the code necessary to implement localization by particle filters.
 * Modify only the sections marked with the TODO comment. 
 */
 
 // Nombre: OSCAR CORTES CALDERON
 
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
	/*
	 * TODO:
	 * Generate a set of N particles (each particle represented by a Pose2D message)
	 * with positions uniformly distributed within bounding box given by min_x, ..., max_a.
	 * To generate uniformly distributed random numbers, you can use the funcion rnd.uniformReal(min, max)
	 */
	
	/*
	 */

        // Generar N partículas con distribución uniforme en la caja [min_x,max_x]x[min_y,max_y] y ángulo [min_a,max_a]
        for (int i = 0; i < N; ++i)
        {
            geometry_msgs::msg::Pose2D p;
            p.x     = rnd.uniformReal(min_x, max_x);
            p.y     = rnd.uniformReal(min_y, max_y);
            p.theta = rnd.uniformReal(min_a, max_a);

            // Normalizar ángulo a (-pi, pi]
            // p.theta = std::atan2(std::sin(p.theta), std::cos(p.theta));

            particles[i] = p;
        }
	
	return particles;
    }

    static void move_particles(std::vector<geometry_msgs::msg::Pose2D>& particles,
			       float delta_x, float delta_y, float delta_t, float sigma2)
    {
	random_numbers::RandomNumberGenerator rnd;
	/*
	 * TODO:
	 * Move each particle a displacement given by delta_x, delta_y and delta_t.
	 * Displacement is given w.r.t. particles's frame, i.e., to calculate the new position for
	 * each particle you need to make a z-rotation of delta_x and delta_y, an angle theta_i, where theta_i
	 * is the orientation of the i-th particle:
	 * xi += delta_x*cos(theta_i) - delta_y*sin(theta_i) + rnd
	 * yi += delta_x*sin(theta_i) + delta_y*cos(theta_i) + rnd
	 * theta_i += delta_t + rnd
	 * Add gaussian noise to each new position. Use sigma2 as variance.
	 * You can use the function rnd.gaussian(mean, variance)
	 */

	/*
	 */
	 
	// Para cada partícula, aplica el desplazamiento (en el marco de la partícula)
        // y agrega ruido gaussiano con varianza sigma2.
        for (auto& p : particles)
        {
            const double c = std::cos(p.theta);
            const double s = std::sin(p.theta);

            // Transformación del desplazamiento del marco de la partícula al mundo
            p.x     +=  delta_x * c - delta_y * s + rnd.gaussian(0.0, sigma2);
            p.y     +=  delta_x * s + delta_y * c + rnd.gaussian(0.0, sigma2);
            p.theta +=  delta_t + rnd.gaussian(0.0, sigma2);

            // Normalizar el ángulo a (-pi, pi]
            p.theta = std::atan2(std::sin(p.theta), std::cos(p.theta));
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
	/*
	 * TODO:
	 * For each particle, calculate the similarity between its simulated scan and the real scan.
	 * Normalize all similarities (the sum of all values must always be 1.0)
	 * Store results in 'similarities'.
	 * IMPORTANT NOTE 1. The real sensor scans are DOWNSAMPLED. That is, only 1 out of 'downsampling' scans is considered.
	 * For example, if downsampling=10, then, if real sensor has 500 ranges, simulated scans will only have 50 ranges
	 * When comparing readings, for each reading in the simulated scan, you should skip 'downsampling' readings
	 * in the real sensor.
	 * IMPORTANT NOTE 2. Both, simulated an real scans, can have infinite distances. Thus, when comparing readings,
	 * ensure both simulated and real ranges are finite values.
	 * Steps:
	 * FOR i=[0... simulated_scans.size())
	 *    delta = 0
	 *    FOR j=[0... simulated_scans[i].ranges.size())
	 *       IF real_scan.ranges[j*downsampling] in valid range AND simulated_scans[i].ranges[j] in valid range:
	 *          delta += |simulated_scans[i].ranges[j] - real_scan.ranges[j*downsampling]|
	 *       ELSE
	 *          delta += max_range
	 *    delta /= simulated_scans[i].ranges.size()
	 *    similarities[i] = exp(-delta/sigma2)
	 *    Normalize all similarities
	 */
	
	/*
	 */
	 
	const double eps = 1e-9;  
        const double max_range = std::isfinite(real_scan.range_max) ? real_scan.range_max : 10.0;

        for (size_t i = 0; i < simulated_scans.size(); ++i)
        {
            const auto& sim_ranges = simulated_scans[i].ranges;
            double delta = 0.0;
            int count = 0;

            for (size_t j = 0; j < sim_ranges.size(); ++j)
            {
                size_t idx_real = j * static_cast<size_t>(downsampling);
                if (idx_real >= real_scan.ranges.size()) break;

                const double s = sim_ranges[j];
                const double r = real_scan.ranges[idx_real];

                if (std::isfinite(s) && std::isfinite(r))
                    delta += std::abs(s - r);
                else
                    delta += max_range;

                ++count;
            }

            if (count > 0) delta /= static_cast<double>(count);

            double like = std::exp(-delta / std::max<double>(sigma2, eps));
            if (!std::isfinite(like)) like = 0.0;

            similarities[i] = like;
        }

        // Normalización a distribución de probabilidad (suma = 1)
        double sum = 0.0;
        for (double v : similarities) sum += v;

        if (sum <= eps || !std::isfinite(sum))
        {
            // fallback uniforme si todo quedó ~0 o NaN
            const double uni = 1.0 / std::max<size_t>(1, similarities.size());
            for (double& v : similarities) v = uni;
        }
        else
        {
            for (double& v : similarities) v /= sum;
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
	
	// Suma total (por si no viene exactamente normalizado a 1.0)
        double sum = 0.0;
        for (double p : probabilities)
            if (std::isfinite(p) && p > 0.0) sum += p;

        if (sum <= 0.0) 
        {
            // Si algo salió mal (todas 0/NaN/inf), no hay forma de elegir con peso;
            // regresamos -1 como indica el esqueleto.
            return -1;
        }

        // x en [0, sum) garantiza correcto muestreo aun si no está normalizado a 1
        double x = rnd.uniformReal(0.0, sum);

        // Algoritmo sugerido por el profe (versión con resta acumulada)
        for (size_t i = 0; i < probabilities.size(); ++i) 
        {
            const double p = (std::isfinite(probabilities[i]) && probabilities[i] > 0.0)
                               ? probabilities[i] : 0.0;

            if (x < p)
                return static_cast<int>(i);
            else
                x -= p;
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
	
	/*
	 */
	 
	const size_t N = particles.size();
        for (size_t k = 0; k < N; ++k)
        {
            // Elige índice según la distribución 'probabilities'
            int idx = random_choice(probabilities);
            if (idx < 0 || static_cast<size_t>(idx) >= N) {
                // Fallback defensivo en caso extremo
                idx = 0;
            }

            // Copia la partícula seleccionada (muestreo con reemplazo)
            geometry_msgs::msg::Pose2D p = particles[static_cast<size_t>(idx)];

            // Agrega ruido gaussiano con varianza sigma2
            p.x     += rnd.gaussian(0.0, sigma2);
            p.y     += rnd.gaussian(0.0, sigma2);
            p.theta += rnd.gaussian(0.0, sigma2);

            // Normaliza ángulo a (-pi, pi]
            p.theta = std::atan2(std::sin(p.theta), std::cos(p.theta));

            resampled_particles[k] = p;
        }
	
	return resampled_particles;
    }
    
};
