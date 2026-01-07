#include "plugin.hpp"
#include <random>

struct QuantumSuperpositionDelay : Module {
	enum ParamId {
		DELAY_TIME_PARAM,
		SPREAD_PARAM,
		PROBABILITY_PARAM,
		FEEDBACK_PARAM,
		MIX_PARAM,
		CHAOS_PARAM,
		PARAMS_LEN
	};
	enum InputId {
		AUDIO_INPUT,
		CV_PROB_INPUT,
		CV_SPREAD_INPUT,
		CV_FEEDBACK_INPUT,
		COLLAPSE_TRIGGER_INPUT,
		INPUTS_LEN
	};
	enum OutputId {
		AUDIO_OUTPUT,
		OUTPUTS_LEN
	};
	enum LightId {
		COLLAPSE_LIGHT,
		BUFFER_LIGHT_1,
		BUFFER_LIGHT_2,
		BUFFER_LIGHT_3,
		BUFFER_LIGHT_4,
		BUFFER_LIGHT_5,
		BUFFER_LIGHT_6,
		LIGHTS_LEN
	};

	static constexpr int NUM_BUFFERS = 6;
	static constexpr int MAX_DELAY_SAMPLES = 96000; // 2 seconds at 48kHz
	static constexpr int BUFFER_SIZE = MAX_DELAY_SAMPLES / NUM_BUFFERS;

	// Delay buffers
	float delayBuffers[NUM_BUFFERS][BUFFER_SIZE];
	int writeIndex = 0;
	int readIndices[NUM_BUFFERS] = {0};

	// Quantum state variables
	float probWeights[NUM_BUFFERS];
	float targetWeights[NUM_BUFFERS];
	float weightVelocity[NUM_BUFFERS];
	float delayTimes[NUM_BUFFERS]; // in samples
	float feedbackLevels[NUM_BUFFERS];
	float entanglement[NUM_BUFFERS];

	// Control variables
	float baseDelayTime = 0.5f; // 0-1 range
	float spreadAmount = 0.5f;
	float probabilityShape = 0.5f;
	float globalFeedback = 0.3f;
	float dryWetMix = 0.5f;
	float chaosAmount = 0.1f;

	// Collapse trigger
	dsp::SchmittTrigger collapseTrigger;
	float collapseLight = 0.f;

	// Random number generator
	std::mt19937 rng;
	std::uniform_real_distribution<float> uniformDist;

	QuantumSuperpositionDelay() {
		config(PARAMS_LEN, INPUTS_LEN, OUTPUTS_LEN, LIGHTS_LEN);
		
		configParam(DELAY_TIME_PARAM, 0.f, 1.f, 0.25f, "Delay Time", " ms", 0.f, 2000.f);
		configParam(SPREAD_PARAM, 0.f, 1.f, 0.5f, "Time Spread", "%", 0.f, 100.f);
		configParam(PROBABILITY_PARAM, 0.f, 1.f, 0.5f, "Probability Shape", "%", 0.f, 100.f);
		configParam(FEEDBACK_PARAM, 0.f, 0.95f, 0.3f, "Feedback", "%", 0.f, 100.f);
		configParam(MIX_PARAM, 0.f, 1.f, 0.5f, "Dry/Wet Mix", "%", 0.f, 100.f);
		configParam(CHAOS_PARAM, 0.f, 1.f, 0.1f, "Chaos Amount", "%", 0.f, 100.f);

		configInput(AUDIO_INPUT, "Audio");
		configInput(CV_PROB_INPUT, "Probability Distribution CV");
		configInput(CV_SPREAD_INPUT, "Time Spread CV");
		configInput(CV_FEEDBACK_INPUT, "Feedback CV");
		configInput(COLLAPSE_TRIGGER_INPUT, "Quantum Collapse Trigger");

		configOutput(AUDIO_OUTPUT, "Audio");

		configLight(COLLAPSE_LIGHT, "Collapse Event");
		for (int i = 0; i < NUM_BUFFERS; i++) {
			configLight(BUFFER_LIGHT_1 + i, string::f("Buffer %d Activity", i + 1));
		}

		// Initialize buffers
		for (int b = 0; b < NUM_BUFFERS; b++) {
			for (int i = 0; i < BUFFER_SIZE; i++) {
				delayBuffers[b][i] = 0.f;
			}
		}

		// Initialize quantum state
		initializeQuantumState();

		// Seed RNG
		rng.seed(std::random_device{}());
		uniformDist = std::uniform_real_distribution<float>(0.f, 1.f);
	}

	void initializeQuantumState() {
		float equalWeight = 1.f / NUM_BUFFERS;
		
		for (int i = 0; i < NUM_BUFFERS; i++) {
			probWeights[i] = equalWeight;
			targetWeights[i] = equalWeight;
			weightVelocity[i] = 0.f;
			delayTimes[i] = 1000.f + (i * 1500.f); // Initial spread in samples
			feedbackLevels[i] = 0.3f;
			entanglement[i] = 0.f;
		}
	}

	float fastRandom() {
		return uniformDist(rng);
	}

	void updateControls() {
		// Read parameters
		float potTime = params[DELAY_TIME_PARAM].getValue();
		float potSpread = params[SPREAD_PARAM].getValue();
		float potProb = params[PROBABILITY_PARAM].getValue();
		float potFeedback = params[FEEDBACK_PARAM].getValue();
		float potMix = params[MIX_PARAM].getValue();
		float potChaos = params[CHAOS_PARAM].getValue();

		// Read CV inputs (0-10V normalized to 0-1)
		float cvProb = inputs[CV_PROB_INPUT].getVoltage() / 10.f;
		float cvSpread = inputs[CV_SPREAD_INPUT].getVoltage() / 10.f;
		float cvFeedback = inputs[CV_FEEDBACK_INPUT].getVoltage() / 10.f;

		// Combine pot + CV
		baseDelayTime = clamp(potTime, 0.f, 1.f);
		spreadAmount = clamp(potSpread + cvSpread, 0.f, 1.f);
		probabilityShape = clamp(potProb + cvProb, 0.f, 1.f);
		globalFeedback = clamp(potFeedback + cvFeedback, 0.f, 0.95f);
		dryWetMix = potMix;
		chaosAmount = potChaos;
	}

	void updateProbabilityWeights() {
		float weights[NUM_BUFFERS];

		if (probabilityShape < 0.5f) {
			// More uniform distribution
			float uniformity = (0.5f - probabilityShape) * 2.f;
			for (int i = 0; i < NUM_BUFFERS; i++) {
				weights[i] = (1.f - uniformity) * targetWeights[i] + uniformity / NUM_BUFFERS;
			}
		} else {
			// More peaked distribution
			float peakedness = (probabilityShape - 0.5f) * 2.f;
			
			static float peakCenter = NUM_BUFFERS / 2.f;
			peakCenter += (fastRandom() - 0.5f) * chaosAmount * 0.5f;
			peakCenter = clamp(peakCenter, 0.f, (float)(NUM_BUFFERS - 1));

			float totalWeight = 0.f;
			for (int i = 0; i < NUM_BUFFERS; i++) {
				float distance = std::abs(i - peakCenter);
				weights[i] = std::exp(-distance * peakedness * 2.f);
				totalWeight += weights[i];
			}

			// Normalize
			for (int i = 0; i < NUM_BUFFERS; i++) {
				weights[i] /= totalWeight;
			}
		}

		// Add chaos
		for (int i = 0; i < NUM_BUFFERS; i++) {
			float chaos = (fastRandom() - 0.5f) * chaosAmount * 0.1f;
			weights[i] = clamp(weights[i] + chaos, 0.f, 1.f);
		}

		// Normalize after chaos
		float sum = 0.f;
		for (int i = 0; i < NUM_BUFFERS; i++) {
			sum += weights[i];
		}
		for (int i = 0; i < NUM_BUFFERS; i++) {
			targetWeights[i] = weights[i] / sum;
		}

		// Smooth interpolation
		for (int i = 0; i < NUM_BUFFERS; i++) {
			float error = targetWeights[i] - probWeights[i];
			weightVelocity[i] = weightVelocity[i] * 0.9f + error * 0.1f;
			probWeights[i] += weightVelocity[i] * 0.05f;
		}
	}

	void updateDelayTimes(float sampleRate) {
		// Convert base delay time from 0-1 to samples
		float minDelaySamples = 10.f; // ~0.2ms minimum
		float maxDelaySamples = (baseDelayTime * 2000.f / 1000.f) * sampleRate; // 0-2000ms
		maxDelaySamples = clamp(maxDelaySamples, minDelaySamples, (float)(BUFFER_SIZE - 1));

		for (int i = 0; i < NUM_BUFFERS; i++) {
			float t = i / (float)(NUM_BUFFERS - 1);
			float delayRange = (maxDelaySamples - minDelaySamples) * spreadAmount;
			delayTimes[i] = minDelaySamples + t * delayRange;

			// Add slight randomization
			delayTimes[i] += (fastRandom() - 0.5f) * sampleRate * 0.005f * chaosAmount;
			delayTimes[i] = clamp(delayTimes[i], 1.f, (float)(BUFFER_SIZE - 1));

			// Calculate read index
			int delaySamples = (int)delayTimes[i];
			readIndices[i] = (writeIndex - delaySamples + BUFFER_SIZE) % BUFFER_SIZE;
		}
	}

	void handleQuantumCollapse() {
		int dominantBuffer = fastRandom() * NUM_BUFFERS;
		float collapseFactor = 0.7f;

		for (int i = 0; i < NUM_BUFFERS; i++) {
			if (i == dominantBuffer) {
				targetWeights[i] = collapseFactor;
			} else {
				targetWeights[i] = (1.f - collapseFactor) / (NUM_BUFFERS - 1);
			}
		}

		collapseLight = 1.f;
	}

	void process(const ProcessArgs& args) override {
		// Update controls periodically
		static int controlDivider = 0;
		if (++controlDivider >= 64) {
			controlDivider = 0;
			updateControls();
			updateProbabilityWeights();
			updateDelayTimes(args.sampleRate);
		}

		// Check for collapse trigger
		if (collapseTrigger.process(inputs[COLLAPSE_TRIGGER_INPUT].getVoltage(), 0.1f, 2.f)) {
			handleQuantumCollapse();
		}

		// Decay collapse light
		collapseLight -= collapseLight / args.sampleRate * 5.f;
		lights[COLLAPSE_LIGHT].setBrightness(collapseLight);

		// Read input
		float inputSample = inputs[AUDIO_INPUT].getVoltage();

		// Write to all delay buffers
		for (int b = 0; b < NUM_BUFFERS; b++) {
			delayBuffers[b][writeIndex] = inputSample;
		}

		// Read from delay buffers with quantum superposition
		float outputAccumulator = 0.f;

		for (int b = 0; b < NUM_BUFFERS; b++) {
			// Read delayed sample with linear interpolation
			int readIdx = readIndices[b];
			float frac = delayTimes[b] - (int)delayTimes[b];
			int readIdxNext = (readIdx + 1) % BUFFER_SIZE;
			
			float delayedSample = delayBuffers[b][readIdx] * (1.f - frac) + 
			                      delayBuffers[b][readIdxNext] * frac;

			// Apply probability weight
			float weightedSample = delayedSample * probWeights[b];
			outputAccumulator += weightedSample;

			// Apply feedback with entanglement
			float feedbackSample = delayedSample * globalFeedback * feedbackLevels[b];

			// Entanglement: feedback influences other buffers
			for (int other = 0; other < NUM_BUFFERS; other++) {
				if (other != b) {
					float entanglementAmount = entanglement[b] * 0.1f;
					int entangleIndex = (writeIndex + 10) % BUFFER_SIZE;
					delayBuffers[other][entangleIndex] += feedbackSample * entanglementAmount;
				}
			}

			// Self-feedback
			delayBuffers[b][writeIndex] += feedbackSample;

			// Update entanglement based on buffer energy
			float energy = std::abs(delayedSample) / 10.f; // Normalize to ~0-1
			entanglement[b] = entanglement[b] * 0.99f + energy * 0.01f;

			// Update buffer activity lights
			lights[BUFFER_LIGHT_1 + b].setBrightness(probWeights[b]);
		}

		// Mix dry and wet
		float wetSample = clamp(outputAccumulator, -10.f, 10.f);
		float mixedOutput = inputSample * (1.f - dryWetMix) + wetSample * dryWetMix;

		// Output
		outputs[AUDIO_OUTPUT].setVoltage(mixedOutput);

		// Advance write pointer
		writeIndex = (writeIndex + 1) % BUFFER_SIZE;
	}

	json_t* dataToJson() override {
		json_t* rootJ = json_object();
		
		// Save quantum state for continuity
		json_t* weightsJ = json_array();
		for (int i = 0; i < NUM_BUFFERS; i++) {
			json_array_append_new(weightsJ, json_real(probWeights[i]));
		}
		json_object_set_new(rootJ, "probWeights", weightsJ);
		
		return rootJ;
	}

	void dataFromJson(json_t* rootJ) override {
		// Restore quantum state
		json_t* weightsJ = json_object_get(rootJ, "probWeights");
		if (weightsJ) {
			for (int i = 0; i < NUM_BUFFERS; i++) {
				json_t* weightJ = json_array_get(weightsJ, i);
				if (weightJ) {
					probWeights[i] = json_real_value(weightJ);
					targetWeights[i] = probWeights[i];
				}
			}
		}
	}
};

struct QuantumSuperpositionDelayWidget : ModuleWidget {
	QuantumSuperpositionDelayWidget(QuantumSuperpositionDelay* module) {
		setModule(module);
		setPanel(createPanel(asset::plugin(pluginInstance, "res/QuantumSuperpositionDelay.svg")));

		addChild(createWidget<ScrewSilver>(Vec(RACK_GRID_WIDTH, 0)));
		addChild(createWidget<ScrewSilver>(Vec(box.size.x - 2 * RACK_GRID_WIDTH, 0)));
		addChild(createWidget<ScrewSilver>(Vec(RACK_GRID_WIDTH, RACK_GRID_HEIGHT - RACK_GRID_WIDTH)));
		addChild(createWidget<ScrewSilver>(Vec(box.size.x - 2 * RACK_GRID_WIDTH, RACK_GRID_HEIGHT - RACK_GRID_WIDTH)));

		// Parameters (left column)
		float knobX = 15.f;
		float knobY = 50.f;
		float knobSpacing = 52.f;

		addParam(createParamCentered<RoundLargeBlackKnob>(mm2px(Vec(knobX, knobY)), module, QuantumSuperpositionDelay::DELAY_TIME_PARAM));
		addParam(createParamCentered<RoundLargeBlackKnob>(mm2px(Vec(knobX, knobY + knobSpacing)), module, QuantumSuperpositionDelay::SPREAD_PARAM));
		addParam(createParamCentered<RoundLargeBlackKnob>(mm2px(Vec(knobX, knobY + knobSpacing * 2)), module, QuantumSuperpositionDelay::PROBABILITY_PARAM));
		addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(knobX, knobY + knobSpacing * 3)), module, QuantumSuperpositionDelay::FEEDBACK_PARAM));
		addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(knobX, knobY + knobSpacing * 3.7)), module, QuantumSuperpositionDelay::MIX_PARAM));
		addParam(createParamCentered<RoundBlackKnob>(mm2px(Vec(knobX, knobY + knobSpacing * 4.4)), module, QuantumSuperpositionDelay::CHAOS_PARAM));

		// CV Inputs (right column)
		float cvX = 40.f;
		float cvY = 30.f;
		float cvSpacing = 20.f;

		addInput(createInputCentered<PJ301MPort>(mm2px(Vec(cvX, cvY)), module, QuantumSuperpositionDelay::AUDIO_INPUT));
		addInput(createInputCentered<PJ301MPort>(mm2px(Vec(cvX, cvY + cvSpacing)), module, QuantumSuperpositionDelay::CV_PROB_INPUT));
		addInput(createInputCentered<PJ301MPort>(mm2px(Vec(cvX, cvY + cvSpacing * 2)), module, QuantumSuperpositionDelay::CV_SPREAD_INPUT));
		addInput(createInputCentered<PJ301MPort>(mm2px(Vec(cvX, cvY + cvSpacing * 3)), module, QuantumSuperpositionDelay::CV_FEEDBACK_INPUT));
		addInput(createInputCentered<PJ301MPort>(mm2px(Vec(cvX, cvY + cvSpacing * 4)), module, QuantumSuperpositionDelay::COLLAPSE_TRIGGER_INPUT));

		// Output
		addOutput(createOutputCentered<PJ301MPort>(mm2px(Vec(cvX, cvY + cvSpacing * 5.5)), module, QuantumSuperpositionDelay::AUDIO_OUTPUT));

		// Lights
		float lightX = 40.f;
		float lightY = 160.f;
		float lightSpacing = 6.f;

		addChild(createLightCentered<MediumLight<RedLight>>(mm2px(Vec(lightX, lightY)), module, QuantumSuperpositionDelay::COLLAPSE_LIGHT));
		
		for (int i = 0; i < QuantumSuperpositionDelay::NUM_BUFFERS; i++) {
			addChild(createLightCentered<SmallLight<BlueLight>>(mm2px(Vec(lightX + (i % 3) * lightSpacing, lightY + 10.f + (i / 3) * lightSpacing)), module, QuantumSuperpositionDelay::BUFFER_LIGHT_1 + i));
		}
	}
};

Model* modelQuantumSuperpositionDelay = createModel<QuantumSuperpositionDelay, QuantumSuperpositionDelayWidget>("QuantumSuperpositionDelay");
