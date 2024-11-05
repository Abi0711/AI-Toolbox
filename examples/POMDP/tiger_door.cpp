/* This file contains the code that is presented in the TutorialPOMDP.md file
 * in the docs folder. The tutorial can also be viewed in the main page of the
 * doxygen documentation.
 *
 * This code implements a problem where the agent, standing in front of two
 * doors, must figure out which of the two is hiding a treasure. The problem is
 * that behind the other door there is a tiger! The agent must thus wait and
 * listen for noise, and try to figure out with enough certainty which door is
 * safe to open.
 *
 * For more examples be sure to check out the "tests" folder! The code there
 * is very simple and it contains most usages of this library ever, and it will
 * probably give you an even better introduction than this code does.
 */
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <thread>
#include <numeric>


#include <AIToolbox/POMDP/Model.hpp>
#include <AIToolbox/MDP/Model.hpp>

#include <AIToolbox/POMDP/Algorithms/SARSOP.hpp>
#include <AIToolbox/POMDP/Algorithms/POMCP.hpp>
#include <AIToolbox/POMDP/Policies/Policy.hpp>
#include <AIToolbox/Tools/CassandraParser.hpp>
#include <fstream>

// RENDERING

// Special character to go back up when drawing.
std::string up =   "\033[XA";
// Special character to go back to the beginning of the line.
std::string back = "\33[2K\r";

void goup(unsigned x) {
    while (x > 9) {
        up[2] = '0' + 9;
        std::cout << up;
        x -= 9;
    }
    up[2] = '0' + x;
    std::cout << up;
}

void godown(unsigned x) {
    while (x) {
        std::cout << '\n';
        --x;
    }
}

const std::vector<std::string> prize {
    { R"(  ________  )" },
    { R"(  |       |\)" },
    { R"(  |_______|/)" },
    { R"( / $$$$  /| )" },
    { R"(+-------+ | )" },
    { R"(|       |/  )" },
    { R"(+-------+   )" }};

const std::vector<std::string> tiger {
    { R"(            )" },
    { R"(   (`/' ` | )" },
    { R"(  /'`\ \   |)" },
    { R"( /<7' ;  \ \)" },
    { R"(/  _､-, `,-\)" },
    { R"(`-`  ､/ ;   )" },
    { R"(     `-'    )" }};

const std::vector<std::string> closedDoor {
    { R"(   ______   )" },
    { R"(  /  ||  \  )" },
    { R"( |   ||   | )" },
    { R"( |   ||   | )" },
    { R"( |   ||   | )" },
    { R"( +===++===+ )" },
    { R"(            )" }};

const std::vector<std::string> openDoor {
    { R"(   ______   )" },
    { R"(|\/      \/|)" },
    { R"(||        ||)" },
    { R"(||        ||)" },
    { R"(||        ||)" },
    { R"(||________||)" },
    { R"(|/        \|)" }};

const std::vector<std::string> sound {
    { R"(    -..-    )" },
    { R"(            )" },
    { R"(  '-,__,-'  )" },
    { R"(            )" },
    { R"( `,_    _,` )" },
    { R"(    `--`    )" },
    { R"(            )" }};

const std::vector<std::string> nosound {
    { R"(            )" },
    { R"(            )" },
    { R"(            )" },
    { R"(            )" },
    { R"(            )" },
    { R"(            )" },
    { R"(            )" }};
// Different format for him!
const std::vector<std::string> man {
    { R"(   ___   )" },
    { R"(  //|\\  )" },
    { R"(  \___/  )" },
    { R"( \__|__/ )" },
    { R"(    |    )" },
    { R"(    |    )" },
    { R"(   / \   )" },
    { R"(  /   \  )" }};

// Random spaces to make the rendering look nice. Yeah this is ugly, but it's
// just for the rendering.
const std::string hspacer{"     "};
const std::string manhspacer(hspacer.size() / 2 + prize[0].size() - man[0].size() / 2, ' ');
const std::string numspacer((prize[0].size() - 8)/2, ' ');

const std::string clockSpacer = numspacer + std::string((hspacer.size() - 1) / 2, ' ');
const std::string strclock(R"(/|\-)");

// MODEL

enum {
    A_LISTEN = 0,
    A_LEFT   = 1,
    A_RIGHT  = 2,
};

enum {
    TIG_LEFT    = 0,
    TIG_RIGHT   = 1,
};

inline AIToolbox::POMDP::Model<AIToolbox::MDP::Model> makeTigerProblem() {
    // Actions are: 0-listen, 1-open-left, 2-open-right
    size_t S = 2, A = 3, O = 2;

    AIToolbox::POMDP::Model<AIToolbox::MDP::Model> model(O, S, A);

    AIToolbox::DumbMatrix3D transitions(boost::extents[S][A][S]);
    AIToolbox::DumbMatrix3D rewards(boost::extents[S][A][S]);
    AIToolbox::DumbMatrix3D observations(boost::extents[S][A][O]);

    // Transitions
    // If we listen, nothing changes.
    for ( size_t s = 0; s < S; ++s )
        transitions[s][A_LISTEN][s] = 1.0;

    // If we pick a door, tiger and treasure shuffle.
    for ( size_t s = 0; s < S; ++s ) {
        for ( size_t s1 = 0; s1 < S; ++s1 ) {
            transitions[s][A_LEFT ][s1] = 1.0 / S;
            transitions[s][A_RIGHT][s1] = 1.0 / S;
        }
    }

    // Observations
    // If we listen, we guess right 85% of the time.
    observations[TIG_LEFT ][A_LISTEN][TIG_LEFT ] = 0.85;
    observations[TIG_LEFT ][A_LISTEN][TIG_RIGHT] = 0.15;

    observations[TIG_RIGHT][A_LISTEN][TIG_RIGHT] = 0.85;
    observations[TIG_RIGHT][A_LISTEN][TIG_LEFT ] = 0.15;

    // Otherwise we get no information on the environment.
    for ( size_t s = 0; s < S; ++s ) {
        for ( size_t o = 0; o < O; ++o ) {
            observations[s][A_LEFT ][o] = 1.0 / O;
            observations[s][A_RIGHT][o] = 1.0 / O;
        }
    }

    // Rewards
    // Listening has a small penalty
    for ( size_t s = 0; s < S; ++s )
        for ( size_t s1 = 0; s1 < S; ++s1 )
            rewards[s][A_LISTEN][s1] = -1.0;

    // Treasure has a decent reward, and tiger a bad penalty.
    for ( size_t s1 = 0; s1 < S; ++s1 ) {
        rewards[TIG_RIGHT][A_LEFT][s1] = 10.0;
        rewards[TIG_LEFT ][A_LEFT][s1] = -100.0;

        rewards[TIG_LEFT ][A_RIGHT][s1] = 10.0;
        rewards[TIG_RIGHT][A_RIGHT][s1] = -100.0;
    }

    model.setTransitionFunction(transitions);
    model.setRewardFunction(rewards);
    model.setObservationFunction(observations);

    return model;
}

// Define the actions and observations
enum {
    A_NORTH = 0,
    A_SOUTH = 1,
    A_EAST  = 2,
    A_WEST  = 3,
    A_SAMPLE = 4,
    A_CHECK1 = 5,
    A_CHECK2 = 6,
    A_CHECK3 = 7,
    A_CHECK4 = 8,
    A_CHECK5 = 9,
    A_CHECK6 = 10,
    A_CHECK7 = 11,
    A_CHECK8 = 12
};

enum {
    O_GOOD = 0,
    O_BAD = 1,
    O_NONE = 2
};

inline AIToolbox::POMDP::Model<AIToolbox::MDP::Model> makeRockSampleProblem() {
    const size_t GRID_SIZE = 11;
    const size_t NUM_ROCKS = 8;
    const size_t S = GRID_SIZE * GRID_SIZE * (1 << NUM_ROCKS); // States: position x rocks
    const size_t A = 13; // north, south, east, west, sample, check1-8
    const size_t O = 3;  // good, bad, none

    AIToolbox::POMDP::Model<AIToolbox::MDP::Model> model(O, S, A);

    AIToolbox::DumbMatrix3D transitions(boost::extents[S][A][S]);
    AIToolbox::DumbMatrix3D rewards(boost::extents[S][A][S]);
    AIToolbox::DumbMatrix3D observations(boost::extents[S][A][O]);

    // Movement success probability
    const double MOVE_PROB = 0.9;

    // Transitions
    for(size_t s = 0; s < S; ++s) {
        // Get position and rock states from state number
        size_t pos = s % (GRID_SIZE * GRID_SIZE);
        size_t rocks = s / (GRID_SIZE * GRID_SIZE);
        size_t x = pos % GRID_SIZE;
        size_t y = pos / GRID_SIZE;

        // Movement actions
        // North
        if(y < GRID_SIZE - 1) 
            transitions[s][A_NORTH][s + GRID_SIZE] = MOVE_PROB;
        transitions[s][A_NORTH][s] = 1 - MOVE_PROB;

        // South
        if(y > 0)
            transitions[s][A_SOUTH][s - GRID_SIZE] = MOVE_PROB;
        transitions[s][A_SOUTH][s] = 1 - MOVE_PROB;

        // East
        if(x < GRID_SIZE - 1)
            transitions[s][A_EAST][s + 1] = MOVE_PROB;
        transitions[s][A_EAST][s] = 1 - MOVE_PROB;

        // West
        if(x > 0)
            transitions[s][A_WEST][s - 1] = MOVE_PROB;
        transitions[s][A_WEST][s] = 1 - MOVE_PROB;

        // Sample action
        transitions[s][A_SAMPLE][s] = 1.0;

        // Check actions
        for(size_t a = A_CHECK1; a <= A_CHECK8; ++a) {
            transitions[s][a][s] = 1.0;
        }
    }

    // Observations
    // For movement and sampling
    for(size_t s = 0; s < S; ++s) {
        for(size_t a = 0; a <= A_SAMPLE; ++a) {
            observations[s][a][O_NONE] = 1.0;
        }
    }

    // For check actions
    const double CHECK_CORRECT_PROB = 0.95;
    for(size_t s = 0; s < S; ++s) {
        for(size_t r = 0; r < NUM_ROCKS; ++r) {
            size_t checkAction = A_CHECK1 + r;
            
            // Get rock state (good/bad)
            bool isGood = (s / (GRID_SIZE * GRID_SIZE)) & (1 << r);
            
            if(isGood) {
                observations[s][checkAction][O_GOOD] = CHECK_CORRECT_PROB;
                observations[s][checkAction][O_BAD] = 1 - CHECK_CORRECT_PROB;
            } else {
                observations[s][checkAction][O_BAD] = CHECK_CORRECT_PROB;
                observations[s][checkAction][O_GOOD] = 1 - CHECK_CORRECT_PROB;
            }
        }
    }

    // Rewards
    // Sample action
    for(size_t s = 0; s < S; ++s) {
        for(size_t s1 = 0; s1 < S; ++s1) {
            // At rock position and rock is good
            if(/* at rock position */ true && /* rock is good */ true)
                rewards[s][A_SAMPLE][s1] = 10.0;
            else
                rewards[s][A_SAMPLE][s1] = -10.0;
        }
    }

    // Exit reward (when reaching rightmost column)
    for(size_t s = 0; s < S; ++s) {
        size_t x = (s % (GRID_SIZE * GRID_SIZE)) % GRID_SIZE;
        if(x == GRID_SIZE - 1) {
            for(size_t s1 = 0; s1 < S; ++s1) {
                rewards[s][A_EAST][s1] = 10.0;
            }
        }
    }

    model.setTransitionFunction(transitions);
    model.setRewardFunction(rewards);
    model.setObservationFunction(observations);

    return model;
}

inline AIToolbox::POMDP::Model<AIToolbox::MDP::Model> make7x7RockSampleProblem() {
    // Smaller version for testing
    const size_t GRID_SIZE = 7;
    const size_t NUM_ROCKS = 5;
    
    const size_t S = GRID_SIZE * GRID_SIZE * (1 << NUM_ROCKS); 
    const size_t A = 10;  // north, south, east, west, sample, check1-5
    const size_t O = 3;  // good, bad, none

    AIToolbox::POMDP::Model<AIToolbox::MDP::Model> model(O, S, A);

    AIToolbox::DumbMatrix3D transitions(boost::extents[S][A][S]);
    AIToolbox::DumbMatrix3D rewards(boost::extents[S][A][S]);
    AIToolbox::DumbMatrix3D observations(boost::extents[S][A][O]);

    const double MOVE_PROB = 0.9;

    // Initialize all transitions to 0
    for(size_t s = 0; s < S; ++s)
        for(size_t a = 0; a < A; ++a)
            for(size_t s1 = 0; s1 < S; ++s1)
                transitions[s][a][s1] = 0.0;

    // Set valid transitions
    for(size_t s = 0; s < S; ++s) {
        size_t pos = s % (GRID_SIZE * GRID_SIZE);
        size_t x = pos % GRID_SIZE;
        size_t y = pos / GRID_SIZE;

        // NORTH
        if(y < GRID_SIZE - 1) {
            transitions[s][A_NORTH][s + GRID_SIZE] = MOVE_PROB;
            transitions[s][A_NORTH][s] = 1.0 - MOVE_PROB;
        } else {
            transitions[s][A_NORTH][s] = 1.0; // Can't move, stay in place
        }

        // SOUTH
        if(y > 0) {
            transitions[s][A_SOUTH][s - GRID_SIZE] = MOVE_PROB;
            transitions[s][A_SOUTH][s] = 1.0 - MOVE_PROB;
        } else {
            transitions[s][A_SOUTH][s] = 1.0;
        }

        // EAST
        if(x < GRID_SIZE - 1) {
            transitions[s][A_EAST][s + 1] = MOVE_PROB;
            transitions[s][A_EAST][s] = 1.0 - MOVE_PROB;
        } else {
            transitions[s][A_EAST][s] = 1.0;
        }

        // WEST
        if(x > 0) {
            transitions[s][A_WEST][s - 1] = MOVE_PROB;
            transitions[s][A_WEST][s] = 1.0 - MOVE_PROB;
        } else {
            transitions[s][A_WEST][s] = 1.0;
        }

        // SAMPLE and CHECK actions - stay in same state
        for(size_t a = A_SAMPLE; a < A; ++a) {
            transitions[s][a][s] = 1.0;
        }
    }

    // Observations
    for(size_t s = 0; s < S; ++s) {
        for(size_t a = 0; a < A; ++a) {
            double sum = 0.0;
            for(size_t o = 0; o < O; ++o) {
                observations[s][a][o] = 0.0;
            }
            
            if(a <= A_SAMPLE) {
                // Movement and sample actions
                observations[s][a][O_NONE] = 1.0;
            } else {
                // Check actions
                size_t rockNum = a - A_SAMPLE - 1;
                bool isGood = (s / (GRID_SIZE * GRID_SIZE)) & (1 << rockNum);
                
                if(isGood) {
                    observations[s][a][O_GOOD] = 0.95;
                    observations[s][a][O_BAD] = 0.05;
                } else {
                    observations[s][a][O_GOOD] = 0.05;
                    observations[s][a][O_BAD] = 0.95;
                }
            }
        }
    }

    // Rewards - all initially 0
    for(size_t s = 0; s < S; ++s) {
        for(size_t a = 0; a < A; ++a) {
            for(size_t s1 = 0; s1 < S; ++s1) {
                rewards[s][a][s1] = 0.0;
            }
        }
    }

    // Set specific rewards
    for(size_t s = 0; s < S; ++s) {
        size_t pos = s % (GRID_SIZE * GRID_SIZE);
        size_t x = pos % GRID_SIZE;
        
        // Exit reward
        if(x == GRID_SIZE - 1) {
            for(size_t s1 = 0; s1 < S; ++s1) {
                rewards[s][A_EAST][s1] = 10.0;
            }
        }

        // Sample rewards will need rock positions defined
        // For now, just penalty for sampling anywhere
        //if(a == A_SAMPLE) {
            for(size_t s1 = 0; s1 < S; ++s1) {
                rewards[s][A_SAMPLE][s1] = -10.0;
            }
        //}
    }

    model.setTransitionFunction(transitions);
    model.setRewardFunction(rewards);
    model.setObservationFunction(observations);

    return model;
}
inline AIToolbox::POMDP::Model<AIToolbox::MDP::Model> POMCPModel(){
    // Create model of the problem.
    std::ifstream fin("rocksample-11-11-parseable.txt");
    AIToolbox::CassandraParser parser;
    auto problem = parser.parsePOMDP(fin);

    AIToolbox::POMDP::Model<AIToolbox::MDP::Model> model(std::get<2>(problem), std::get<0>(problem), std::get<1>(problem));

    model.setTransitionFunction(std::get<3>(problem));
    model.setRewardFunction(std::get<4>(problem));
    model.setObservationFunction(std::get<5>(problem));
    return model;

}
inline AIToolbox::POMDP::Model<AIToolbox::MDP::Model> make9x9RockSampleProblem() {
    const size_t GRID_SIZE = 3;
    const size_t NUM_ROCKS = 8;
    const size_t S = GRID_SIZE * GRID_SIZE * (1 << NUM_ROCKS);
    const size_t A = 13;  // north, south, east, west, sample, check1-8
    const size_t O = 3;   // good, bad, none
    const double MOVE_PROB = 0.9;

    AIToolbox::POMDP::Model<AIToolbox::MDP::Model> model(O, S, A);
    AIToolbox::DumbMatrix3D transitions(boost::extents[S][A][S]);
    AIToolbox::DumbMatrix3D rewards(boost::extents[S][A][S]);
    AIToolbox::DumbMatrix3D observations(boost::extents[S][A][O]);

    // Initialize movement actions and their transitions
    auto setMovement = [&](size_t s, size_t action, size_t nextState) {
        transitions[s][action][nextState] = MOVE_PROB;
        transitions[s][action][s] = 1.0 - MOVE_PROB;
    };

    for(size_t s = 0; s < S; ++s) {
        size_t pos = s % (GRID_SIZE * GRID_SIZE);
        size_t x = pos % GRID_SIZE;
        size_t y = pos / GRID_SIZE;

        // Set default transitions for all actions in current state
        std::fill_n(&transitions[s][0][0], A * S, 0.0);
        
        // Movement transitions
        if(y < GRID_SIZE - 1) setMovement(s, A_NORTH, s + GRID_SIZE);
        else transitions[s][A_NORTH][s] = 1.0;
        
        if(y > 0) setMovement(s, A_SOUTH, s - GRID_SIZE);
        else transitions[s][A_SOUTH][s] = 1.0;
        
        if(x < GRID_SIZE - 1) setMovement(s, A_EAST, s + 1);
        else transitions[s][A_EAST][s] = 1.0;
        
        if(x > 0) setMovement(s, A_WEST, s - 1);
        else transitions[s][A_WEST][s] = 1.0;

        // Static actions (SAMPLE and CHECK)
        for(size_t a = A_SAMPLE; a < A; ++a) {
            transitions[s][a][s] = 1.0;
        }

        // Set observations
        std::fill_n(&observations[s][0][0], A * O, 0.0);
        
        // Movement and sample actions always observe 'none'
        for(size_t a = 0; a <= A_SAMPLE; ++a) {
            observations[s][a][O_NONE] = 1.0;
        }

        // Check actions observations
        for(size_t a = A_SAMPLE + 1; a < A; ++a) {
            bool isGood = (s / (GRID_SIZE * GRID_SIZE)) & (1 << (a - A_SAMPLE - 1));
            observations[s][a][O_GOOD] = isGood ? 0.95 : 0.05;
            observations[s][a][O_BAD] = isGood ? 0.05 : 0.95;
        }

        // Set rewards
        std::fill_n(&rewards[s][0][0], A * S, 0.0);
        
        // Exit reward
        if(x == GRID_SIZE - 1) {
            std::fill_n(&rewards[s][A_EAST][0], S, 10.0);
        }
        
        // Sample penalty
        std::fill_n(&rewards[s][A_SAMPLE][0], S, -10.0);
    }

    model.setTransitionFunction(transitions);
    model.setRewardFunction(rewards);
    model.setObservationFunction(observations);
    model.setDiscount(0.95);
    return model;
}
void chronoExample() {
    // Start time point
    auto start = std::chrono::high_resolution_clock::now();
    
    // Your code here
    // ... do something ...
    
    // End time point
    auto end = std::chrono::high_resolution_clock::now();
    
    // Calculate duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Output results
    std::cout << "Time taken: " << duration.count() << " microseconds" << std::endl;
    
    // If you want milliseconds instead:
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time taken: " << ms.count() << " milliseconds" << std::endl;
}

inline AIToolbox::POMDP::Model<AIToolbox::MDP::Model> make9x9RockSampleProblemWithDistance() {
    const size_t GRID_SIZE = 3;
    const size_t NUM_ROCKS = 8;
    const size_t S = GRID_SIZE * GRID_SIZE * (1 << NUM_ROCKS);
    const size_t A = 13;  // north, south, east, west, sample, check1-8
    const size_t O = 3;   // good, bad, none
    const double MOVE_PROB = 0.9;

    // Rock positions (x,y coordinates)
    const std::vector<std::pair<int, int>> ROCK_POSITIONS = {
        {0,0}, {1,0}, {2,0},
        {0,1}, {1,1}, {2,1},
        {0,2}, {1,2}
    };

    AIToolbox::POMDP::Model<AIToolbox::MDP::Model> model(O, S, A);
    AIToolbox::DumbMatrix3D transitions(boost::extents[S][A][S]);
    AIToolbox::DumbMatrix3D rewards(boost::extents[S][A][S]);
    AIToolbox::DumbMatrix3D observations(boost::extents[S][A][O]);

    // Calculate sensor efficiency based on distance
    auto calculateEfficiency = [](int rx, int ry, int x, int y) {
        double distance = std::sqrt(std::pow(rx - x, 2) + std::pow(ry - y, 2));
        // Exponential decay of efficiency with distance
        return std::exp(-0.5 * distance);
    };

    // Initialize movement actions and their transitions
    auto setMovement = [&](size_t s, size_t action, size_t nextState) {
        transitions[s][action][nextState] = MOVE_PROB;
        transitions[s][action][s] = 1.0 - MOVE_PROB;
    };

    for(size_t s = 0; s < S; ++s) {
        size_t pos = s % (GRID_SIZE * GRID_SIZE);
        size_t x = pos % GRID_SIZE;
        size_t y = pos / GRID_SIZE;

        // Set default transitions for all actions in current state
        std::fill_n(&transitions[s][0][0], A * S, 0.0);
        
        // Movement transitions
        if(y < GRID_SIZE - 1) setMovement(s, A_NORTH, s + GRID_SIZE);
        else transitions[s][A_NORTH][s] = 1.0;
        
        if(y > 0) setMovement(s, A_SOUTH, s - GRID_SIZE);
        else transitions[s][A_SOUTH][s] = 1.0;
        
        if(x < GRID_SIZE - 1) setMovement(s, A_EAST, s + 1);
        else transitions[s][A_EAST][s] = 1.0;
        
        if(x > 0) setMovement(s, A_WEST, s - 1);
        else transitions[s][A_WEST][s] = 1.0;

        // Static actions (SAMPLE and CHECK)
        for(size_t a = A_SAMPLE; a < A; ++a) {
            transitions[s][a][s] = 1.0;
        }

        // Set observations
        std::fill_n(&observations[s][0][0], A * O, 0.0);
        
        // Movement and sample actions always observe 'none'
        for(size_t a = 0; a <= A_SAMPLE; ++a) {
            observations[s][a][O_NONE] = 1.0;
        }

        // Check actions observations with distance-based efficiency
        for(size_t a = A_SAMPLE + 1; a < A; ++a) {
            size_t rockIndex = a - A_SAMPLE - 1;
            bool isGood = (s / (GRID_SIZE * GRID_SIZE)) & (1 << rockIndex);
            
            // Calculate sensor efficiency based on distance to rock
            double efficiency = calculateEfficiency(
                ROCK_POSITIONS[rockIndex].first,
                ROCK_POSITIONS[rockIndex].second,
                x, y
            );

            // Linear interpolation between perfect sensing (η=1) and random (η=0)
            double correctProb = 0.5 + 0.5 * efficiency;  // ranges from 0.5 to 1.0
            double incorrectProb = 1.0 - correctProb;     // ranges from 0.5 to 0.0

            observations[s][a][O_GOOD] = isGood ? correctProb : incorrectProb;
            observations[s][a][O_BAD] = isGood ? incorrectProb : correctProb;
        }

        // Set rewards
        std::fill_n(&rewards[s][0][0], A * S, 0.0);
        
        // Exit reward
        if(x == GRID_SIZE - 1) {
            std::fill_n(&rewards[s][A_EAST][0], S, 10.0);
        }
        
        // Sample penalty
        std::fill_n(&rewards[s][A_SAMPLE][0], S, -10.0);
    }

    model.setTransitionFunction(transitions);
    model.setRewardFunction(rewards);
    model.setObservationFunction(observations);
    model.setDiscount(0.95);
    return model;
}

inline AIToolbox::POMDP::Model<AIToolbox::MDP::Model> make9x9RockSampleProblemEXAMPLE() {
    std::cout << "jgkh" << std::endl;

    const size_t GRID_SIZE = 4;
    const size_t NUM_ROCKS = 4;
    const size_t S = GRID_SIZE * GRID_SIZE * std::pow(3, NUM_ROCKS);
    const size_t A = 9;  // north, south, east, west, sample, check1-4
    const size_t O = 3;   // good, bad, none
    const double MOVE_PROB = 0.9;

    // Corrected to match NUM_ROCKS = 4
    const std::vector<std::pair<int, int>> ROCK_POSITIONS = {
        {0,0}, {1,0}, {2,0}, {0,1}  // Only 4 rock positions
    };
    std::cout << "jgkh" << std::endl;

    AIToolbox::POMDP::Model<AIToolbox::MDP::Model> model(O, S, A);
    std::cout << "jgkh" << std::endl;
    AIToolbox::DumbMatrix3D transitions(boost::extents[S][A][S]);
    AIToolbox::DumbMatrix3D rewards(boost::extents[S][A][S]);
    AIToolbox::DumbMatrix3D observations(boost::extents[S][A][O]);
    std::cout << "jgkh" << std::endl;


    // Helper functions for state encoding
    auto getRockState = [](size_t s, size_t rockIndex) {
        size_t positionBits = 25;  // GRID_SIZE * GRID_SIZE
        size_t shifted = s / positionBits;
        for(size_t i = 0; i < rockIndex; ++i) {
            shifted /= 3;  // Move to next rock's state
        }
        return shifted % 3;
    };

    auto setRockSampled = [](size_t s, size_t rockIndex) {
        size_t positionBits = 25;  // GRID_SIZE * GRID_SIZE
        size_t multiplier = std::pow(3, rockIndex);
        size_t rockMask = 2 * multiplier * positionBits;
        return (s & ~rockMask) | rockMask;
    };


    auto getNearestRock = [&](size_t x, size_t y) -> int {
        int nearestRock = -1;
        double minDist = std::numeric_limits<double>::max();
        
        for(size_t i = 0; i < NUM_ROCKS; ++i) {  // Changed to NUM_ROCKS
            double dist = std::sqrt(
                std::pow(ROCK_POSITIONS[i].first - static_cast<int>(x), 2) + 
                std::pow(ROCK_POSITIONS[i].second - static_cast<int>(y), 2)
            );
            if(dist < minDist) {
                minDist = dist;
                nearestRock = i;
            }
        }
        return minDist < 0.1 ? nearestRock : -1;
    };

    for(size_t s = 0; s < S; ++s) {
        size_t pos = s % (GRID_SIZE * GRID_SIZE);
        size_t x = pos % GRID_SIZE;
        size_t y = pos / GRID_SIZE;
        

        // Movement transitions
        size_t nextState;
        if(y < GRID_SIZE - 1) {
            nextState = s + GRID_SIZE;
            transitions[s][0][nextState] = MOVE_PROB;
            transitions[s][0][s] = 1.0 - MOVE_PROB;
        } else {
            transitions[s][0][s] = 1.0;
        }
        std::cout<<"passed 1"<<std::endl;
        
        if(y > 0) {
            nextState = s - GRID_SIZE;
            transitions[s][1][nextState] = MOVE_PROB;
            transitions[s][1][s] = 1.0 - MOVE_PROB;
        } else {
            transitions[s][1][s] = 1.0;
        }
        std::cout<<"passed 2"<< std::endl;
        if(x < GRID_SIZE - 1) {
            nextState = s + 1;
            transitions[s][2][nextState] = MOVE_PROB;
            transitions[s][2][s] = 1.0 - MOVE_PROB;
        } else {
            transitions[s][2][s] = 1.0;
        }
        
        if(x > 0) {
            nextState = s - 1;
            transitions[s][3][nextState] = MOVE_PROB;
            transitions[s][3][s] = 1.0 - MOVE_PROB;
        } else {
            transitions[s][3][s] = 1.0;
        }
        std::cout<<"passed 1"<< std::endl;
        // Sample action (action 4)
        int nearestRock = getNearestRock(x, y);
        if(nearestRock >= 0) {
            size_t rockState = getRockState(s, nearestRock);
            if(rockState != 2) {
                size_t nextState = setRockSampled(s, nearestRock);
                transitions[s][4][nextState] = 1.0;
                rewards[s][4][nextState] = (rockState == 1) ? 10.0 : -10.0;
            } else {
                transitions[s][4][s] = 1.0;
                rewards[s][4][s] = -10.0;
            }
        } else {
            transitions[s][4][s] = 1.0;
            rewards[s][4][s] = -10.0;
        }

        // Movement and sample actions observe 'none'
        for(size_t a = 0; a <= 4; ++a) {
            observations[s][a][2] = 1.0;  // O_NONE = 2
        }

        // Check actions (5-8)
        for(size_t a = 5; a < A; ++a) {
            transitions[s][a][s] = 1.0;
            size_t rockIndex = a - 5;
            size_t rockState = getRockState(s, rockIndex);
            
            if(rockState == 2) {
                observations[s][a][2] = 1.0;  // O_NONE = 2
                continue;
            }

            double dist = std::sqrt(
                std::pow(ROCK_POSITIONS[rockIndex].first - static_cast<int>(x), 2) + 
                std::pow(ROCK_POSITIONS[rockIndex].second - static_cast<int>(y), 2)
            );
            double efficiency = std::exp(-0.5 * dist);
            double correctProb = 0.5 + 0.5 * efficiency;
            double incorrectProb = 1.0 - correctProb;

            observations[s][a][0] = (rockState == 1) ? correctProb : incorrectProb;  // O_GOOD = 0
            observations[s][a][1] = (rockState == 1) ? incorrectProb : correctProb;  // O_BAD = 1
        }

        // Exit reward
        if(x == GRID_SIZE - 1) {
            for(size_t sp = 0; sp < S; ++sp) {
                rewards[s][2][sp] = 10.0;  // A_EAST = 2
            }
        }
    }

    model.setTransitionFunction(transitions);
    model.setRewardFunction(rewards);
    model.setObservationFunction(observations);
    model.setDiscount(0.95);
    return model;
}

inline AIToolbox::POMDP::Model<AIToolbox::MDP::Model> testerf() {
    std::cout << "jgkh" << std::endl;
    const size_t GRID_SIZE = 5;
    const size_t NUM_ROCKS = 4;
    const size_t S = GRID_SIZE * GRID_SIZE * std::pow(3, NUM_ROCKS);
    const size_t A = 9;  // north, south, east, west, sample, check1-4
    const size_t O = 3;   // good, bad, none
    const double MOVE_PROB = 0.9;
    
    AIToolbox::POMDP::Model<AIToolbox::MDP::Model> model(O, S, A);

    // Fixed rock positions for 4 rocks
    const std::vector<std::pair<int, int>> ROCK_POSITIONS = {
        {0,0}, {1,0}, {2,0}, {0,1}
    };
    std::cout << "jgjkh" << std::endl;

    AIToolbox::DumbMatrix3D transitions(boost::extents[S][A][S]);
    AIToolbox::DumbMatrix3D rewards(boost::extents[S][A][S]);
    AIToolbox::DumbMatrix3D observations(boost::extents[S][A][O]);

    // Fast state manipulation functions
    auto getRockState = [](size_t s, size_t rockIndex) {
        return (s / (25 * static_cast<size_t>(std::pow(3, rockIndex)))) % 3;
    };

    auto setRockSampled = [](size_t s, size_t rockIndex) {
        const size_t positionBits = 25;
        const size_t power = std::pow(3, rockIndex);
        const size_t mask = ~(2ULL * power * positionBits);
        return (s & mask) | (2ULL * power * positionBits);
    };

    // Fast lookup for distances
    std::vector<std::vector<double>> rockEfficiencies(GRID_SIZE * GRID_SIZE, 
                                                     std::vector<double>(NUM_ROCKS));
    std::cout << "look" << std::endl;
    
    for(size_t pos = 0; pos < GRID_SIZE * GRID_SIZE; ++pos) {
        int x = pos % GRID_SIZE;
        int y = pos / GRID_SIZE;
        for(size_t r = 0; r < NUM_ROCKS; ++r) {
            double dist = std::sqrt(
                std::pow(ROCK_POSITIONS[r].first - x, 2) + 
                std::pow(ROCK_POSITIONS[r].second - y, 2)
            );
            rockEfficiencies[pos][r] = std::exp(-0.5 * dist);
        }
    }
    std::cout << "start" << std::endl;
    
    for(size_t s = 0; s < S; ++s) {
        size_t pos = s % (GRID_SIZE * GRID_SIZE);
        size_t x = pos % GRID_SIZE;
        size_t y = pos / GRID_SIZE;

        // Movement actions (0-3)
        if(y < GRID_SIZE - 1) {
            transitions[s][0][s + GRID_SIZE] = MOVE_PROB;
            transitions[s][0][s] = 1.0 - MOVE_PROB;
            observations[s][0][2] = 1.0;
        } else {
            transitions[s][0][s] = 1.0;
            observations[s][0][2] = 1.0;
        }
        
        if(y > 0) {
            transitions[s][1][s - GRID_SIZE] = MOVE_PROB;
            transitions[s][1][s] = 1.0 - MOVE_PROB;
            observations[s][1][2] = 1.0;
        } else {
            transitions[s][1][s] = 1.0;
            observations[s][1][2] = 1.0;
        }
        
        if(x < GRID_SIZE - 1) {
            transitions[s][2][s + 1] = MOVE_PROB;
            transitions[s][2][s] = 1.0 - MOVE_PROB;
            observations[s][2][2] = 1.0;
            if(x == GRID_SIZE - 2) {  // About to exit
                rewards[s][2][s + 1] = 10.0;
            }
        } else {
            transitions[s][2][s] = 1.0;
            observations[s][2][2] = 1.0;
        }
        
        if(x > 0) {
            transitions[s][3][s - 1] = MOVE_PROB;
            transitions[s][3][s] = 1.0 - MOVE_PROB;
            observations[s][3][2] = 1.0;
        } else {
            transitions[s][3][s] = 1.0;
            observations[s][3][2] = 1.0;
        }
    std::cout << s << std::endl;

        // Sample action (4)
        transitions[s][4][s] = 1.0;
        observations[s][4][2] = 1.0;
        bool canSample = false;
        
        // Check if we're at any rock position
        for(size_t r = 0; r < NUM_ROCKS; ++r) {
            if(x == ROCK_POSITIONS[r].first && y == ROCK_POSITIONS[r].second) {
                size_t rockState = getRockState(s, r);
                if(rockState != 2) {  // Not sampled
                    transitions[s][4][setRockSampled(s, r)] = 1.0;
                    transitions[s][4][s] = 0.0;
                    rewards[s][4][setRockSampled(s, r)] = (rockState == 1) ? 10.0 : -10.0;
                    canSample = true;
                }
                break;
            }
        }
        
        if(!canSample) {
            rewards[s][4][s] = -10.0;
        }

        // Check actions (5-8)
        for(size_t r = 0; r < NUM_ROCKS; ++r) {
            size_t a = r + 5;
            transitions[s][a][s] = 1.0;
            
            size_t rockState = getRockState(s, r);
            if(rockState == 2) {
                observations[s][a][2] = 1.0;  // Already sampled
                continue;
            }

            double efficiency = rockEfficiencies[pos][r];
            double correctProb = 0.5 + 0.5 * efficiency;
            
            observations[s][a][0] = (rockState == 1) ? correctProb : 1.0 - correctProb;
            observations[s][a][1] = (rockState == 1) ? 1.0 - correctProb : correctProb;
        }
    }

    model.setTransitionFunction(transitions);
    model.setRewardFunction(rewards);
    model.setObservationFunction(observations);
    model.setDiscount(0.95);

    return model;
}
inline AIToolbox::POMDP::Model<AIToolbox::MDP::Model> testerplease() {
    const size_t GRID_SIZE = 5;
    const size_t NUM_ROCKS = 4;
    const size_t S = GRID_SIZE * GRID_SIZE * std::pow(3, NUM_ROCKS);
    const size_t A = 9;
    const size_t O = 3;
    const double MOVE_PROB = 0.9;
    
    AIToolbox::POMDP::Model<AIToolbox::MDP::Model> model(O, S, A);

    const std::vector<std::pair<int, int>> ROCK_POSITIONS = {
        {0,0}, {1,0}, {2,0}, {0,1}
    };

    AIToolbox::DumbMatrix3D transitions(boost::extents[S][A][S]);
    AIToolbox::DumbMatrix3D rewards(boost::extents[S][A][S]);
    AIToolbox::DumbMatrix3D observations(boost::extents[S][A][O]);

    // Initialize all matrices to 0
    for(size_t s = 0; s < S; ++s) {
        for(size_t a = 0; a < A; ++a) {
            for(size_t sp = 0; sp < S; ++sp) {
                transitions[s][a][sp] = 0.0;
                rewards[s][a][sp] = 0.0;
            }
            for(size_t o = 0; o < O; ++o) {
                observations[s][a][o] = 0.0;
            }
            observations[s][a][2] = 1.0;  // Default to 'none' observation
        }
    }

    auto getRockState = [](size_t s, size_t rockIndex) {
        return (s / (25 * static_cast<size_t>(std::pow(3, rockIndex)))) % 3;
    };

    auto setRockSampled = [](size_t s, size_t rockIndex) {
        const size_t positionBits = 25;
        const size_t power = std::pow(3, rockIndex);
        const size_t mask = ~(2ULL * power * positionBits);
        return (s & mask) | (2ULL * power * positionBits);
    };

    // Precompute rock efficiencies
    std::vector<std::vector<double>> rockEfficiencies(GRID_SIZE * GRID_SIZE, 
                                                     std::vector<double>(NUM_ROCKS));
    for(size_t pos = 0; pos < GRID_SIZE * GRID_SIZE; ++pos) {
        int x = pos % GRID_SIZE;
        int y = pos / GRID_SIZE;
        for(size_t r = 0; r < NUM_ROCKS; ++r) {
            double dist = std::sqrt(
                std::pow(ROCK_POSITIONS[r].first - x, 2) + 
                std::pow(ROCK_POSITIONS[r].second - y, 2)
            );
            rockEfficiencies[pos][r] = std::exp(-0.5 * dist);
        }
    }
    
    for(size_t s = 0; s < S; ++s) {
        size_t pos = s % (GRID_SIZE * GRID_SIZE);
        size_t x = pos % GRID_SIZE;
        size_t y = pos / GRID_SIZE;

        // North
        if(y < GRID_SIZE - 1) {
            transitions[s][0][s + GRID_SIZE] = MOVE_PROB;
            transitions[s][0][s] = 1.0 - MOVE_PROB;
        } else {
            transitions[s][0][s] = 1.0;
        }

        // South
        if(y > 0) {
            transitions[s][1][s - GRID_SIZE] = MOVE_PROB;
            transitions[s][1][s] = 1.0 - MOVE_PROB;
        } else {
            transitions[s][1][s] = 1.0;
        }

        // East
        if(x < GRID_SIZE - 1) {
            transitions[s][2][s + 1] = MOVE_PROB;
            transitions[s][2][s] = 1.0 - MOVE_PROB;
            if(x == GRID_SIZE - 2) {
                rewards[s][2][s + 1] = 10.0;
            }
        } else {
            transitions[s][2][s] = 1.0;
        }

        // West
        if(x > 0) {
            transitions[s][3][s - 1] = MOVE_PROB;
            transitions[s][3][s] = 1.0 - MOVE_PROB;
        } else {
            transitions[s][3][s] = 1.0;
        }

        // Sample action
        bool sampledRock = false;
        for(size_t r = 0; r < NUM_ROCKS; ++r) {
            if(x == ROCK_POSITIONS[r].first && y == ROCK_POSITIONS[r].second) {
                size_t rockState = getRockState(s, r);
                if(rockState != 2) {  // Not sampled
                    size_t nextState = setRockSampled(s, r);
                    transitions[s][4][nextState] = 1.0;
                    rewards[s][4][nextState] = (rockState == 1) ? 10.0 : -10.0;
                    sampledRock = true;
                }
                break;
            }
        }
        if(!sampledRock) {
            transitions[s][4][s] = 1.0;
            rewards[s][4][s] = -10.0;
        }

        // Check actions
        for(size_t r = 0; r < NUM_ROCKS; ++r) {
            size_t a = r + 5;
            transitions[s][a][s] = 1.0;  // Stay in same state
            
            size_t rockState = getRockState(s, r);
            if(rockState == 2) {
                observations[s][a][2] = 1.0;  // None observation if sampled
                continue;
            }

            double efficiency = rockEfficiencies[pos][r];
            double correctProb = 0.5 + 0.5 * efficiency;
            
            if(rockState == 1) {  // Good rock
                observations[s][a][0] = correctProb;     // Good observation
                observations[s][a][1] = 1.0 - correctProb; // Bad observation
            } else {  // Bad rock
                observations[s][a][0] = 1.0 - correctProb; // Good observation
                observations[s][a][1] = correctProb;     // Bad observation
            }
        }
    }

    // Validate transition probabilities
    for(size_t s = 0; s < S; ++s) {
        for(size_t a = 0; a < A; ++a) {
            double sum = 0.0;
            for(size_t sp = 0; sp < S; ++sp) {
                sum += transitions[s][a][sp];
            }
            if(std::abs(sum - 1.0) > 1e-9) {
                std::cout << "Invalid transition probabilities for state " << s 
                         << " action " << a << " sum: " << sum << std::endl;
            }
        }
    }

    model.setTransitionFunction(transitions);
    model.setRewardFunction(rewards);
    model.setObservationFunction(observations);
    model.setDiscount(0.95);

    return model;
}

inline double POMCProck(unsigned horizon){
    std::cout << "jgkh" << std::endl;

    AIToolbox::POMDP::Model m = testerplease();
    std::cout<<"made model"<<"\n";
    //     //for 7x5
    // const size_t GRID_SIZE = 7;
    // const size_t NUM_ROCKS = 5;
    //     // POMCP Parameters
    // // State space is 7x7x2^4 = 49x16 = 784 states (much smaller than 11x11x2^8)

    // size_t beliefSize = 1500;      // Fewer particles needed due to smaller state space
    // unsigned iterations = 1000;     // Fewer iterations needed for smaller space
    // double explorationConstant = 20.0;  // Rewards still [-10,10] but less complexity
    // unsigned horizon = 50;         // Smaller grid means shorter paths to goal

            //for 9x8
    const size_t GRID_SIZE = 4;
    const size_t NUM_ROCKS = 4;
        // POMCP Parameters
    size_t beliefSize = 1500;       // More particles for better belief representation
    unsigned iterations = 1500;      // More thorough search
    double explorationConstant = 25.0;
    
    //const size_t S = GRID_SIZE * GRID_SIZE * (1 << NUM_ROCKS); 
     const size_t S = GRID_SIZE * GRID_SIZE * std::pow(3, NUM_ROCKS);  // Use base-3 for rock states

    AIToolbox::POMDP::Belief initialBelief(S);
    initialBelief.fill(1.0/S);  // Each state has equal probability
    // Random engine for sampling
    std::default_random_engine rand(AIToolbox::Seeder::getSeed());

    // Sample initial state from belief
    auto initialState = AIToolbox::sampleProbability(S, initialBelief, rand);
    std::cout<<"l"<<"\n";



    AIToolbox::POMDP::POMCP solver(m,
                                beliefSize,
                                iterations,
                                explorationConstant);
    std::cout<<"made sookve"<<"\n";
    

    // Main planning loop
    auto currentState = initialState;
    auto nextState=currentState;
    size_t observation;
    double reward=0.0;
    double totalReward = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int t = horizon ; t > 0; --t) {
        std::cout<<t<<" t\n";

        // Get action from POMCP
        auto action = solver.sampleAction(initialBelief, t);
        // Execute action in environment and get observation
        // This would come from your actual environment
        std::tie(nextState, observation, reward) = m.sampleSOR(currentState, action);
        totalReward += reward;

        // Update solver with action-observation pair
        solver.sampleAction(action, observation, t);
        
        currentState = nextState;
        std::cout << totalReward << "rew \n";

    }
    
    
    // End time point
    auto end = std::chrono::high_resolution_clock::now();
    
    // Calculate duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Output results
    std::cout << "Time taken: " << duration.count() << " microseconds" << std::endl;
    
    // If you want milliseconds instead:
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time taken: " << ms.count() << " milliseconds" << std::endl;
    std::cout << totalReward << "fin\n";
    return totalReward;
}

int main() {
    // We create a random engine, since we will need this later.
    //AIToolbox::POMDP::Model model = POMCPModel();

    std::vector<double> rewards;
    double total = 0.0;

    for (int i = 0; i < 15; ++i) {
        double reward = POMCProck(100);
        rewards.push_back(reward);
        total += reward;
        
        // Calculate and print current average
        double current_avg = total / (i + 1);
        std::cout << "Iteration " << i + 1 << ":\nRewards list: [";
        for (size_t j = 0; j <= i; ++j) {
            std::cout << std::fixed << std::setprecision(2) << rewards[j];
            if (j < i) std::cout << ", ";
        }
        std::cout << "]\nCurrent Average: " << current_avg << "\n\n";
    }

    // Calculate and print final average
    double final_average = std::accumulate(rewards.begin(), rewards.end(), 0.0) / rewards.size();
    std::cout << "\nFinal Results:\nAll rewards: [";
    for (size_t i = 0; i < rewards.size(); ++i) {
        std::cout << std::fixed << std::setprecision(2) << rewards[i];
        if (i < rewards.size() - 1) std::cout << ", ";
    }
    std::cout << "]\nFinal Average: " << final_average << std::endl;
    // Set the horizon. This will determine the optimality of the policy
    // dependent on how many steps of observation/action we plan to do. 1 means
    // we're just going to do one thing only, and we're done. 2 means we get to
    // do a single action, observe the result, and act again. And so on.
    unsigned horizon = 15;
    // The 0.0 is the tolerance factor, used with high horizons. It gives a way
    // to stop the computation if the policy has converged to something static.
    // AIToolbox::POMDP::SARSOP solver(0.0, 0.0001);

    // Solve the model. After this line, the problem has been completely
    // solved. All that remains is setting up an experiment and see what
    // happens!
    // AIToolbox::POMDP::Belief initi(2); initi << 0.5, 0.5;
    // auto solution = solver(model, initi);
    //solution = The lower and upper gap bounds, the lower bound VList, and the upper bound QFunction.

    // We create a policy from the solution, in order to obtain actual actions
    // depending on what happens in the environment.
    //num state, num action, num observations
    // AIToolbox::POMDP::Policy policy(2, 3, 2);

    // We begin a simulation, we start from a uniform belief, which means that
    // we have no idea on which side the tiger is in. We sample from the belief
    // in order to get a "real" state for the world, since this code has to
    // both emulate the environment and control the agent. The agent won't know
    // the sampled state though, it will only have the belief to work with.
    // AIToolbox::POMDP::Belief b(2); b << 0.5, 0.5;
    // auto s = AIToolbox::sampleProbability(2, b, rand);

    // The first thing that happens is that we take an action, so we sample it now.
    // auto [a, ID] = policy.sampleAction(b, horizon);

    // We loop for each step we have yet to do.
    // double totalReward = 0.0;
    // for (int t = horizon - 1; t >= 0; --t) {
    //     // We advance the world one step (the agent only sees the observation
    //     // and reward).
    //     auto [s1, o, r] = model.sampleSOR(s, a);
    //     // We and update our total reward.
    //     totalReward += r;

    //     { // Rendering of the environment, depends on state, action and observation.
    //         auto & left  = s ? prize : tiger;
    //         auto & right = s ? tiger : prize;
    //         for (size_t i = 0; i < prize.size(); ++i)
    //             std::cout << left[i] << hspacer << right[i] << '\n';

    //         auto & dleft  = a == A_LEFT  ? openDoor : closedDoor;
    //         auto & dright = a == A_RIGHT ? openDoor : closedDoor;
    //         for (size_t i = 0; i < prize.size(); ++i)
    //             std::cout << dleft[i] << hspacer << dright[i] << '\n';

    //         auto & sleft  = a == A_LISTEN && o == TIG_LEFT  ? sound : nosound;
    //         auto & sright = a == A_LISTEN && o == TIG_RIGHT ? sound : nosound;
    //         for (size_t i = 0; i < prize.size(); ++i)
    //             std::cout << sleft[i] << hspacer << sright[i] << '\n';

    //         std::cout << numspacer << b[0] << clockSpacer
    //                   << strclock[t % strclock.size()]
    //                   << clockSpacer << b[1] << '\n';

    //         for (const auto & m : man)
    //             std::cout << manhspacer << m << '\n';

    //         std::cout << "Timestep missing: " << t << "       \n";
    //         std::cout << "Total reward:     " << totalReward << "       " << std::endl;

    //         goup(3 * prize.size() + man.size() + 3);
    //     }

    //     // We explicitly update the belief to show the user what the agent is
    //     // thinking. This is also necessary in some cases (depending on
    //     // convergence of the solution, see below), otherwise its only for
    //     // rendering purpouses. It is a pretty expensive operation so if
    //     // performance is required it should be avoided.
    //     b = AIToolbox::POMDP::updateBelief(model, b, a, o);

    //     // Now that we have rendered, we can use the observation to find out
    //     // what action we should do next.
    //     //
    //     // Depending on whether the solution converged or not, we have to use
    //     // the policy differently. Suppose that we planned for an horizon of 5,
    //     // but the solution converged after 3. Then the policy will only be
    //     // usable with horizons of 3 or less. For higher horizons, the highest
    //     // step of the policy suffices (since it converged), but it will need a
    //     // manual belief update to know what to do.
    //     //
    //     // Otherwise, the policy implicitly tracks the belief via the id it
    //     // returned from the last sampling, without the need for a belief
    //     // update. This is a consequence of the fact that POMDP policies are
    //     // computed from a piecewise linear and convex value function, so
    //     // ranges of similar beliefs actually result in needing to do the same
    //     // thing (since they are similar enough for the timesteps considered).
    //     if (t > (int)policy.getH())
    //         std::tie(a, ID) = policy.sampleAction(b, policy.getH());
    //     else
    //         std::tie(a, ID) = policy.sampleAction(ID, o, t);

    //     // Then we update the world
    //     s = s1;

    //     // Sleep 1 second so the user can see what is happening.
    //     std::this_thread::sleep_for(std::chrono::seconds(1));
    // }
    // // Put the cursor back where it should be.
    // godown(3 * prize.size() + man.size() + 3);

    return 0;
}
