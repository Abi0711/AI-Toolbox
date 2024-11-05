#ifndef AI_TOOLBOX_MDP_ROLLOUT_HEADER_FILE
#define AI_TOOLBOX_MDP_ROLLOUT_HEADER_FILE

#include <AIToolbox/TypeTraits.hpp>
#include <AIToolbox/MDP/Types.hpp>

namespace AIToolbox::MDP {
    /**
     * @brief This function performs a rollout from the input state.
     *
     * This function performs a rollout until the agent either reaches the
     * desired depth, or reaches a terminal state. The overall return is
     * finally returned, from the point of the input state, and with the future
     * rewards discounted appropriately.
     *
     * This function is generally used in Monte-Carlo tree search-like
     * algorithms, like MCTS or POMCP, to speed up discovery of promising
     * actions without necessarily expanding their search tree. This avoids
     * wasting lots of computation and memory on states far from our root that
     * we will probably never see again, while at the same time still getting
     * an estimate for the rest of the simulation.
     *
     * @param m The model to use for the rollout.
     * @param s The state to start the rollout from.
     * @param maxDepth The maximum number of timesteps to look into the future.
     * @param rnd A random number generator.
     *
     * @return The discounted return from the input state.
     */
    template <typename M, typename Gen>
    requires AIToolbox::IsGenerativeModel<M> && HasIntegralActionSpace<M>
    double rollout(const M & m, std::remove_cvref_t<decltype(std::declval<M>().getS())> s, const unsigned maxDepth, Gen & rnd) {
        double rew = 0.0, totalRew = 0.0, gamma = 1.0;

        // Here we have two separate branches depending on whether the model
        // provides a variable number of actions or not. Note that they are
        // nearly identical, and the only difference is how we instantiate the
        // distribution from which to randomly sample actions.

        if constexpr (HasFixedActionSpace<M>) {
            // If we don't have variable actions, we instantiate the
            // distribution outside of the loop for a slight performance
            // increase.
            std::uniform_int_distribution<size_t> dist(0, m.getA()-1);
            for (unsigned depth = 0; depth < maxDepth; ++depth ) {

                std::tie( s, rew ) = m.sampleSR( s, dist(rnd) );
                totalRew += gamma * rew;

                if (m.isTerminal(s))
                    return totalRew;

                gamma *= m.getDiscount();
            }
        } else {
            // Otherwise, we need to poll the model at every timestep to check
            // the allowed number of actions, and sample from those.
            for (unsigned depth = 0; depth < maxDepth; ++depth ) {
                std::uniform_int_distribution<size_t> dist(0, m.getA(s)-1);

                std::tie( s, rew ) = m.sampleSR( s, dist(rnd) );
                totalRew += gamma * rew;

                if (m.isTerminal(s))
                    return totalRew;

                gamma *= m.getDiscount();
            }
        }
        return totalRew;
    }


    /**
     * @brief This function performs an adaptive rollout from the input state.
     *
     * This function performs a rollout until one of three conditions is met:
     * 1. The agent reaches the maximum depth
     * 2. The agent reaches a terminal state
     * 3. The rewards have converged within a threshold (adaptive termination)
     *
     * The function uses a sliding window to track recent rewards and can terminate
     * early if the values have stabilized, saving computation in long horizons.
     *
     * @param m The model to use for the rollout.
     * @param s The state to start the rollout from.
     * @param maxDepth The maximum number of timesteps to look into the future.
     * @param rnd A random number generator.
     * @param minDepth The minimum depth before considering early termination.
     * @param windowSize The size of the sliding window for reward convergence.
     * @param threshold The convergence threshold for early termination.
     *
     * @return The discounted return from the input state.
     */
    template <typename M, typename Gen>
    requires AIToolbox::IsGenerativeModel<M> && HasIntegralActionSpace<M>
    double adaptiveRollout(const M & m, 
                        std::remove_cvref_t<decltype(std::declval<M>().getS())> s, 
                        const unsigned maxDepth, 
                        Gen & rnd,
                        const unsigned minDepth = 10,
                        const unsigned windowSize = 5,
                        const double threshold = 0.01) {
        double rew = 0.0, totalRew = 0.0, gamma = 1.0;
        
        // Sliding window for value convergence check
        std::deque<double> rewardWindow;
        double windowSum = 0.0;

        if constexpr (HasFixedActionSpace<M>) {
            std::uniform_int_distribution<size_t> dist(0, m.getA()-1);
            for (unsigned depth = 0; depth < maxDepth; ++depth) {
                std::tie(s, rew) = m.sampleSR(s, dist(rnd));
                totalRew += gamma * rew;

                // Update sliding window
                rewardWindow.push_back(gamma * rew);
                windowSum += gamma * rew;
                if (rewardWindow.size() > windowSize) {
                    windowSum -= rewardWindow.front();
                    rewardWindow.pop_front();
                }

                // Check for early termination conditions
                if (m.isTerminal(s))
                    return totalRew;
                
                // Only check convergence after minimum depth
                if (depth >= minDepth && rewardWindow.size() == windowSize) {
                    double avgReward = windowSum / windowSize;
                    double latestReward = rewardWindow.back();
                    if (std::abs(latestReward - avgReward) < threshold) {
                        return totalRew;
                    }
                }

                gamma *= m.getDiscount();
            }
        } else {
            for (unsigned depth = 0; depth < maxDepth; ++depth) {
                std::uniform_int_distribution<size_t> dist(0, m.getA(s)-1);
                std::tie(s, rew) = m.sampleSR(s, dist(rnd));
                totalRew += gamma * rew;

                // Update sliding window
                rewardWindow.push_back(gamma * rew);
                windowSum += gamma * rew;
                if (rewardWindow.size() > windowSize) {
                    windowSum -= rewardWindow.front();
                    rewardWindow.pop_front();
                }

                // Check for early termination conditions
                if (m.isTerminal(s)) 
                    return totalRew;
                
                // Only check convergence after minimum depth
                if (depth >= minDepth && rewardWindow.size() == windowSize) {
                    double avgReward = windowSum / windowSize;
                    double latestReward = rewardWindow.back();
                    if (std::abs(latestReward - avgReward) < threshold) {
                        return totalRew;
                    }
                }

                gamma *= m.getDiscount();
            }
        }
        return totalRew;
    }
}

#endif
