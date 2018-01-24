/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	normal_distribution<double> x_distr(x, std[0]);
	normal_distribution<double> y_distr(y, std[1]);
	normal_distribution<double> theta_distr(theta, std[2]);
	default_random_engine g;
	num_particles = 100;
	Particle particle;
	for (unsigned i=0; i<num_particles; i++){
		particle.id = i;
		particle.x = x_distr(g);
		particle.y = y_distr(g);
		particle.theta = theta_distr(g);
		particle.weight = 1.0;
		particles.push_back(particle);
		weights.push_back(particle.weight);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    
    static unsigned seed1 = 0;
    ++seed1;
    default_random_engine g (seed1);
    if (yaw_rate > 0.001){
        for (unsigned i=0; i<particles.size(); i++){
            normal_distribution<double> x_distr(particles[i].x + velocity/yaw_rate*(sin(particles[i].theta+yaw_rate*delta_t)-sin(particles[i].theta)), std_pos[0]);
            particles[i].x = x_distr(g);
            normal_distribution<double> y_distr(particles[i].y + velocity/yaw_rate*(cos(particles[i].theta)-cos(particles[i].theta+yaw_rate*delta_t)), std_pos[1]);
            particles[i].y = y_distr(g);
            normal_distribution<double> theta_distr(particles[i].theta + yaw_rate*delta_t, std_pos[2]);
            particles[i].theta = theta_distr(g);
            //if (particles[i].theta > M_PI) particles[i].theta = particles[i].theta - 2*M_PI;
            //if (particles[i].theta < -M_PI) particles[i].theta = particles[i].theta + 2*M_PI;
        }
    }
    else{
        for (unsigned i=0; i<particles.size(); i++){
            normal_distribution<double> x_distr(particles[i].x + velocity/yaw_rate*(sin(particles[i].theta+yaw_rate*delta_t)-sin(particles[i].theta)), std_pos[0]);
            particles[i].x = x_distr(g);
            normal_distribution<double> y_distr(particles[i].y + velocity/yaw_rate*(cos(particles[i].theta)-cos(particles[i].theta+yaw_rate*delta_t)), std_pos[1]);
            particles[i].y = y_distr(g);
            normal_distribution<double> theta_distr(particles[i].theta + yaw_rate*delta_t, std_pos[2]);
            particles[i].theta = theta_distr(g);
            //if (particles[i].theta > M_PI) particles[i].theta = particles[i].theta - 2*M_PI;
            //if (particles[i].theta < -M_PI) particles[i].theta = particles[i].theta + 2*M_PI;
        }
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	//loop over every observation
	for (unsigned z=0; z<observations.size(); z++){
		//loop over every landmark in sensor range with respect to each observations
        std::vector<double> eucl_dist;
		for (unsigned j=0; j<predicted.size(); j++){
			eucl_dist.push_back(dist(predicted[j].x, predicted[j].y, observations[z].x, observations[z].y));
		}
        std::vector<double>::iterator smallest = min_element(eucl_dist.begin(),eucl_dist.end());//id is the predicted vector's index;
        observations[z].id = std::distance(eucl_dist.begin(), smallest);//get index of vector of landmarks in sensor range with respect to each particle
        //cout<<"observation:"<<z<<" "<<"associated with landmark "<<predicted[observations[z].id].id<<endl;
        //eucl_dist.clear();
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    double sum = 0.0;
    for (unsigned i=0; i<particles.size(); i++){
        //get the landmark within sensor range with respect to one particle.
        std::vector<LandmarkObs> predicted_obs;
        for (unsigned j=0; j<map_landmarks.landmark_list.size(); j++){
            LandmarkObs pred_obs_temp;
            double landmark_x = double (map_landmarks.landmark_list[j].x_f);
            double landmark_y = double (map_landmarks.landmark_list[j].y_f);
            if (dist(particles[i].x, particles[i].y, landmark_x, landmark_y)<sensor_range){
                pred_obs_temp.id = map_landmarks.landmark_list[j].id_i;
                pred_obs_temp.x = landmark_x;
                pred_obs_temp.y = landmark_y;
                //cout<<"landmark id:"<<pred_obs_temp.id<<" "<<"landmark x:"<<landmark_x<<" "<<"landmark y:"<<landmark_y<<endl;
                predicted_obs.push_back(pred_obs_temp);
            }
        }
        //measurements transformed to map coordinate system
        std::vector<LandmarkObs> transform_obs;
        for (unsigned z=0; z<observations.size(); z++){
            LandmarkObs t_obs_temp;
            //t_obs_temp.id = observations[z].id;
            t_obs_temp.x = particles[i].x + observations[z].x*cos(particles[i].theta) - observations[z].y*sin(particles[i].theta);
            t_obs_temp.y = particles[i].y + observations[z].x*sin(particles[i].theta) + observations[z].y*cos(particles[i].theta);
            //cout<<"observation x:"<<t_obs_temp.x<<" "<<"observation y:"<<t_obs_temp.y<<endl;
            transform_obs.push_back(t_obs_temp);
        }
        //data associations
        dataAssociation(predicted_obs, transform_obs);
        //particle's data associations
        for (unsigned z=0; z<transform_obs.size(); z++){
            particles[i].associations.push_back(predicted_obs[transform_obs[z].id].id);
            particles[i].sense_x.push_back(transform_obs[z].x);
            particles[i].sense_y.push_back(transform_obs[z].y);
        }
        //update weight of particles using gaussian multivariant distribution
        for (unsigned z=0; z<transform_obs.size(); z++){
            double mu_x = predicted_obs[transform_obs[z].id].x;
            double mu_y = predicted_obs[transform_obs[z].id].y;
            particles[i].weight *= 1.0/(2*M_PI*std_landmark[0]*std_landmark[1])*exp(-(transform_obs[z].x-mu_x)*(transform_obs[z].x-mu_x)/(2*std_landmark[0]*std_landmark[0])-(transform_obs[z].y-mu_y)*(transform_obs[z].y-mu_y)/(2*std_landmark[1]*std_landmark[1]));
        }
        cout<<"particles["<<i<<"]"<<" weight:"<<particles[i].weight<<endl;
        sum +=particles[i].weight;
    }
    for (unsigned j=0; j<particles.size(); j++){
        weights[j] = particles[j].weight/sum;
        particles[j].weight = weights[j];
    }
    //double max_w =*max_element(weights.begin(), weights.end());
    //cout<<"weight max:"<<max_w<<endl;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::vector<int> index;
	for (unsigned j=0; j<particles.size(); j++){
        index.push_back(j);
	}
    
    std::vector<Particle> p;
    double max_w =*max_element(weights.begin(), weights.end());
    cout<<"weight max:"<<max_w<<endl;
    double beta = 0.0;
    std::discrete_distribution<int> index_distr(index.begin(), index.end());
    static unsigned seed = 0;
    seed++;
    std::default_random_engine g (seed);
    int ind = index_distr(g);
    cout<<"random index:"<<ind<<endl;
    for (unsigned j=0; j<particles.size(); j++){
        beta += (index_distr(g)+1)*2*max_w/double(particles.size());
        cout<<"beta:"<<beta<<endl;
        while (weights[ind] < beta){
            beta -= weights[ind];
            ind = (ind + 1) % num_particles;
        }
        cout<<"particle index:"<<ind<<" and weights:"<<particles[ind].weight<<endl;
        p.push_back(particles[ind]);
    }
    particles = p;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
