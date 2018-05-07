/**
 * Author: Liu Chaoyang
 * E-mail: chaoyanglius@gmail.com
 * 
 * histgrams equalize using C++
 * Copyright (C) 2018 Liu Chaoyang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
 
#ifndef CPU_HISTOGRAM_H
#define CPU_HISTOGRAM_H

#include <opencv2/core/core.hpp>

void hist_equal(const cv::Mat &src, cv::Mat &dst);
// implement histogram match from target image
void hist_match(const cv::Mat &src, cv::Mat &dst, const cv::Mat &tgt);
// implement histogram match from target histogram
void hist_match(const cv::Mat &src, cv::Mat &dst, const double hgram[]);

#endif
