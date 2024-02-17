#pragma once

#include <chrono>
#include <cstdio>
#include <map>
#include <vector>
#include <queue>
#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>

class PiraTimer
{
private:
	class Stats {
	private:
		// <total_duration, count>
		std::pair<double, size_t> averages;
		// <duration>
		std::priority_queue<double> medians;
		double calculated_median = 0.0;

	public:
		std::chrono::high_resolution_clock::time_point start_time;

	public:
		void insert_duration(double duration) {
			averages.first += duration;
			averages.second += 1;

			medians.push(duration);
		}

		// TODO: test get_median with even and odd values
		// NOTE: Modifies existing median data, so can only be calculated once
		void calculate_median() {
			// Could happen when start was called, but end wasn't, or it was misspelled
			if (medians.size() == 0)
			{
				calculated_median = 0;
				return;
			}

			int64_t count = medians.size() / 2;
			double last_value = 0;
			while (count > 0) {
				last_value = medians.top();
				medians.pop();
				--count;
			}

			if (medians.size() % 2 == 0) {
				calculated_median = (last_value + medians.top()) / 2;
			}
			else {
				calculated_median = medians.top();
			}
		}

		double get_average() {
			// can happen, when timer was started, but was never ended
			if (averages.second == 0)
				return 0;

			return averages.first / (double)averages.second;
		}

		double get_total() {
			// can happen, when timer was started, but was never ended
			if (averages.second == 0)
				return 0;

			return averages.first;
		}

		double get_median() {
			return calculated_median;
		}

		size_t get_count() {
			return averages.second;
		}
	};

private:
	// Some settings
	static const int decimal_precision = 3;
	static const int spaces_between_words = 3;

	// Some duration stats that are added in end() function
	static inline std::map<std::string, Stats> stats;

public:
	static void start(const std::string& description) {
		stats[description].start_time = std::chrono::high_resolution_clock::now();
	}

	static std::chrono::duration<double, std::milli> end(const std::string& description) {
		if (stats.find(description) != stats.end()) {
			std::chrono::duration<double, std::milli> duration = std::chrono::high_resolution_clock::now() - stats[description].start_time;
			stats[description].insert_duration(duration.count());
			return duration;
		}
			
		// otherwise description wasn't met in "start()" function yet
		return std::chrono::duration<double, std::milli>(0);
	}

	static void print_stats() {
		std::cout << "-------------------------\n";

		auto get_num_len = [](double num) -> size_t {
			return num <= 2.0 ? 1 : log10(num) + 1;
		};

		double total_ms = 0;
		size_t max_description_length = 0;
		size_t max_average_length = 0;
		size_t max_median_length = 0;
		size_t max_total_length = 0;
		for (auto it = stats.begin(); it != stats.end(); ++it) {
			total_ms += it->second.get_average();
			it->second.calculate_median();

			max_description_length = std::max(max_description_length, it->first.length());
			max_average_length = std::max(max_average_length, get_num_len(it->second.get_average()));
			max_median_length = std::max(max_median_length, get_num_len(it->second.get_median()));
			max_total_length = std::max(max_total_length, get_num_len(it->second.get_total()));
		}

		std::cout << std::fixed << std::setprecision(decimal_precision);
		for (auto it = stats.begin(); it != stats.end(); ++it) {
			std::cout << it->first << std::string(max_description_length - it->first.length() + spaces_between_words, ' ');

			std::cout << "Average: " << it->second.get_average() << " ms" << std::string(max_average_length - get_num_len(it->second.get_average()) + spaces_between_words, ' ');
			std::cout << "Median: " << it->second.get_median() << " ms" << std::string(max_median_length - get_num_len(it->second.get_median()) + spaces_between_words, ' ');
			std::cout << "Total: " << it->second.get_total() << " ms" << std::string(max_total_length - get_num_len(it->second.get_total()) + spaces_between_words, ' ');
			std::cout << "Total measured: " << it->second.get_count() << '\n';
		}

		std::cout << "-------------------------\n";
	}

	static void reset_stats() {
		stats.clear();
	}
};