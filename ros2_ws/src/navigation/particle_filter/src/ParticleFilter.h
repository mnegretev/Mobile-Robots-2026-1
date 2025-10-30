/*
 * MOBILE ROBOTS - UNAM, FI, 2026-1
 * LOCALIZATION BY PARTICLE FILTERS
 *
 * Instructions:
 * Write the code necessary to implement localization by particle filters.
 * Modify only the sections marked with the TODO comment. 
 */
#include "particle_filter/ray_tracer.h"
#include <cmath>
#include <numeric>
#include <limits>

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
	
	for(int i = 0; i < N; ++i)
	{
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

	for(size_t i = 0; i < particles.size(); ++i)
	{
	    double theta = particles[i].theta;
	    double nx = rnd.gaussian(0.0, sigma2);
	    double ny = rnd.gaussian(0.0, sigma2);
	    double ntheta = rnd.gaussian(0.0, sigma2);

	    double dx_world = delta_x * std::cos(theta) - delta_y * std::sin(theta);
	    double dy_world = delta_x * std::sin(theta) + delta_y * std::cos(theta);

	    particles[i].x += static_cast<float>(dx_world + nx);
	    particles[i].y += static_cast<float>(dy_world + ny);
	    particles[i].theta += static_cast<float>(delta_t + ntheta);
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
	
	const float max_range = real_scan.range_max > 0.0f ? real_scan.range_max : 1.0f;

	for(size_t i = 0; i < simulated_scans.size(); ++i)
	{
	    const auto &sim = simulated_scans[i];
	    double delta = 0.0;
	    size_t M = sim.ranges.size();
	    if(M == 0)
	    {
		similarities[i] = 0.0;
		continue;
	    }

	    for(size_t j = 0; j < M; ++j)
	    {
		size_t real_idx = j * static_cast<size_t>(downsampling);
		double sim_r = sim.ranges[j];
		double real_r = std::numeric_limits<double>::infinity();
		if(real_idx < real_scan.ranges.size())
		    real_r = real_scan.ranges[real_idx];

		bool sim_finite = std::isfinite(sim_r);
		bool real_finite = std::isfinite(real_r);

		if(sim_finite && real_finite)
		{
		    delta += std::fabs(sim_r - real_r);
		}
		else
		{
		    delta += static_cast<double>(max_range);
		}
	    }

	    delta /= static_cast<double>(M);

	    // per TODO: similarities[i] = exp(-delta/sigma2)
	    similarities[i] = std::exp(-delta / static_cast<double>(sigma2));
	}

	// Normalize
	double sum = std::accumulate(similarities.begin(), similarities.end(), 0.0);
	if(sum <= 0.0)
	{
	    // Avoid division by zero: assign uniform small probability
	    double uniform = 1.0 / static_cast<double>(similarities.size());
	    for(size_t i = 0; i < similarities.size(); ++i) similarities[i] = uniform;
	}
	else
	{
	    for(size_t i = 0; i < similarities.size(); ++i) similarities[i] /= sum;
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
	
	if(probabilities.empty()) return -1;

	double x = rnd.uniformReal(0.0, 1.0);
	// numerical safety: ensure sum may be ~1 but not exactly
	for(size_t i = 0; i < probabilities.size(); ++i)
	{
	    if(x < probabilities[i]) return static_cast<int>(i);
	    x -= probabilities[i];
	    if(x <= 0.0) return static_cast<int>(i); // safety fallback
	}
	// If we get here due to rounding errors, return last index
	return static_cast<int>(probabilities.size() - 1);
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
	
	if(particles.empty() || probabilities.empty()) return resampled_particles;

	for(size_t k = 0; k < resampled_particles.size(); ++k)
	{
	    int idx = random_choice(probabilities);
	    if(idx < 0) idx = 0;
	    geometry_msgs::msg::Pose2D p = particles[static_cast<size_t>(idx)];

	    // Add gaussian noise (mean 0, variance sigma2)
	    double nx = rnd.gaussian(0.0, sigma2);
	    double ny = rnd.gaussian(0.0, sigma2);
	    double ntheta = rnd.gaussian(0.0, sigma2);

	    p.x = static_cast<float>(p.x + nx);
	    p.y = static_cast<float>(p.y + ny);
	    p.theta = static_cast<float>(p.theta + ntheta);

	    resampled_particles[k] = p;
	}
	
	return resampled_particles;
    }
    
};
