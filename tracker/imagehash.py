from __future__ import (absolute_import, division, print_function)

from PIL import Image, ImageFilter
import numpy
__version__ = "4.2.1"


def _binary_array_to_hex(arr):
	"""
	internal function to make a hex string out of a binary array.
	"""
	bit_string = ''.join(str(b) for b in 1 * arr.flatten())
	width = int(numpy.ceil(len(bit_string)/4))
	return '{:0>{width}x}'.format(int(bit_string, 2), width=width)


class ImageHash(object):
	"""
	Hash encapsulation. Can be used for dictionary keys and comparisons.
	"""
	def __init__(self, binary_array):
		self.hash = binary_array

	def __str__(self):
		return _binary_array_to_hex(self.hash.flatten())

	def __repr__(self):
		return repr(self.hash)

	def __sub__(self, other):
		if other is None:
			raise TypeError('Other hash must not be None.')
		if self.hash.size != other.hash.size:
			raise TypeError('ImageHashes must be of the same shape.', self.hash.shape, other.hash.shape)
		return numpy.count_nonzero(self.hash.flatten() != other.hash.flatten())

	def __eq__(self, other):
		if other is None:
			return False
		return numpy.array_equal(self.hash.flatten(), other.hash.flatten())

	def __ne__(self, other):
		if other is None:
			return False
		return not numpy.array_equal(self.hash.flatten(), other.hash.flatten())

	def __hash__(self):
		# this returns a 8 bit integer, intentionally shortening the information
		return sum([2**(i % 8) for i, v in enumerate(self.hash.flatten()) if v])

	def __len__(self):
		# Returns the bit length of the hash
		return self.hash.size


def hex_to_hash(hexstr):
	hash_size = int(numpy.sqrt(len(hexstr)*4))
	#assert hash_size == numpy.sqrt(len(hexstr)*4)
	binary_array = '{:0>{width}b}'.format(int(hexstr, 16), width = hash_size * hash_size)
	bit_rows = [binary_array[i:i+hash_size] for i in range(0, len(binary_array), hash_size)]
	hash_array = numpy.array([[bool(int(d)) for d in row] for row in bit_rows])
	return ImageHash(hash_array)


def hex_to_flathash(hexstr, hashsize):
	hash_size = int(len(hexstr)*4 / (hashsize))
	binary_array = '{:0>{width}b}'.format(int(hexstr, 16), width=hash_size * hashsize)
	hash_array = numpy.array([[bool(int(d)) for d in binary_array]])[-hash_size * hashsize:]
	return ImageHash(hash_array)



def old_hex_to_hash(hexstr, hash_size=8):
	l = []
	count = hash_size * (hash_size // 4)
	if len(hexstr) != count:
		emsg = 'Expected hex string size of {}.'
		raise ValueError(emsg.format(count))
	for i in range(count // 2):
		h = hexstr[i*2:i*2+2]
		v = int("0x" + h, 16)
		l.append([v & 2**i > 0 for i in range(8)])
	return ImageHash(numpy.array(l))


def average_hash(image, hash_size=8, mean=numpy.mean):
	if hash_size < 2:
		raise ValueError("Hash size must be greater than or equal to 2")

	# reduce size and complexity, then covert to grayscale
	image = image.convert("L").resize((hash_size, hash_size), Image.ANTIALIAS)

	# find average pixel value; 'pixels' is an array of the pixel values, ranging from 0 (black) to 255 (white)
	pixels = numpy.asarray(image)
	avg = mean(pixels)

	# create string of bits
	diff = pixels > avg
	# make a hash
	return ImageHash(diff)


def phash(image, hash_size=8, highfreq_factor=4):
	if hash_size < 2:
		raise ValueError("Hash size must be greater than or equal to 2")

	import scipy.fftpack
	img_size = hash_size * highfreq_factor
	image = image.convert("L").resize((img_size, img_size), Image.ANTIALIAS)
	pixels = numpy.asarray(image)
	dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels, axis=0), axis=1)
	dctlowfreq = dct[:hash_size, :hash_size]
	med = numpy.median(dctlowfreq)
	diff = dctlowfreq > med
	return ImageHash(diff)


def phash_simple(image, hash_size=8, highfreq_factor=4):
	import scipy.fftpack
	img_size = hash_size * highfreq_factor
	image = image.convert("L").resize((img_size, img_size), Image.ANTIALIAS)
	pixels = numpy.asarray(image)
	dct = scipy.fftpack.dct(pixels)
	dctlowfreq = dct[:hash_size, 1:hash_size+1]
	avg = dctlowfreq.mean()
	diff = dctlowfreq > avg
	return ImageHash(diff)


def dhash(image, hash_size=8):
	if hash_size < 2:
		raise ValueError("Hash size must be greater than or equal to 2")

	image = image.convert("L").resize((hash_size + 1, hash_size), Image.ANTIALIAS)
	pixels = numpy.asarray(image)
	# compute differences between columns
	diff = pixels[:, 1:] > pixels[:, :-1]
	return ImageHash(diff)


def dhash_vertical(image, hash_size=8):
	# resize(w, h), but numpy.array((h, w))
	image = image.convert("L").resize((hash_size, hash_size + 1), Image.ANTIALIAS)
	pixels = numpy.asarray(image)
	# compute differences between rows
	diff = pixels[1:, :] > pixels[:-1, :]
	return ImageHash(diff)


def whash(image, hash_size = 8, image_scale = None, mode = 'haar', remove_max_haar_ll = True):
	import pywt
	if image_scale is not None:
		assert image_scale & (image_scale - 1) == 0, "image_scale is not power of 2"
	else:
		image_natural_scale = 2**int(numpy.log2(min(image.size)))
		image_scale = max(image_natural_scale, hash_size)

	ll_max_level = int(numpy.log2(image_scale))

	level = int(numpy.log2(hash_size))
	assert hash_size & (hash_size-1) == 0, "hash_size is not power of 2"
	assert level <= ll_max_level, "hash_size in a wrong range"
	dwt_level = ll_max_level - level

	image = image.convert("L").resize((image_scale, image_scale), Image.ANTIALIAS)
	pixels = numpy.asarray(image) / 255.

	# Remove low level frequency LL(max_ll) if @remove_max_haar_ll using haar filter
	if remove_max_haar_ll:
		coeffs = pywt.wavedec2(pixels, 'haar', level = ll_max_level)
		coeffs = list(coeffs)
		coeffs[0] *= 0
		pixels = pywt.waverec2(coeffs, 'haar')

	# Use LL(K) as freq, where K is log2(@hash_size)
	coeffs = pywt.wavedec2(pixels, mode, level = dwt_level)
	dwt_low = coeffs[0]

	# Substract median and compute hash
	med = numpy.median(dwt_low)
	diff = dwt_low > med
	return ImageHash(diff)



def colorhash(image, binbits=3):

	# bin in hsv space:
	intensity = numpy.asarray(image.convert("L")).flatten()
	h, s, v = [numpy.asarray(v).flatten() for v in image.convert("HSV").split()]
	# black bin
	mask_black = intensity < 256 // 8
	frac_black = mask_black.mean()
	# gray bin (low saturation, but not black)
	mask_gray = s < 256 // 3
	frac_gray = numpy.logical_and(~mask_black, mask_gray).mean()
	# two color bins (medium and high saturation, not in the two above)
	mask_colors = numpy.logical_and(~mask_black, ~mask_gray)
	mask_faint_colors = numpy.logical_and(mask_colors, s < 256 * 2 // 3)
	mask_bright_colors = numpy.logical_and(mask_colors, s > 256 * 2 // 3)

	c = max(1, mask_colors.sum())
	# in the color bins, make sub-bins by hue
	hue_bins = numpy.linspace(0, 255, 6+1)
	if mask_faint_colors.any():
		h_faint_counts, _ = numpy.histogram(h[mask_faint_colors], bins=hue_bins)
	else:
		h_faint_counts = numpy.zeros(len(hue_bins) - 1)
	if mask_bright_colors.any():
		h_bright_counts, _ = numpy.histogram(h[mask_bright_colors], bins=hue_bins)
	else:
		h_bright_counts = numpy.zeros(len(hue_bins) - 1)

	# now we have fractions in each category (6*2 + 2 = 14 bins)
	# convert to hash and discretize:
	maxvalue = 2**binbits
	values = [min(maxvalue-1, int(frac_black * maxvalue)), min(maxvalue-1, int(frac_gray * maxvalue))]
	for counts in list(h_faint_counts) + list(h_bright_counts):
		values.append(min(maxvalue-1, int(counts * maxvalue * 1. / c)))
	# print(values)
	bitarray = []
	for v in values:
		bitarray += [v // (2**(binbits-i-1)) % 2**(binbits-i) > 0 for i in range(binbits)]
	return ImageHash(numpy.asarray(bitarray).reshape((-1, binbits)))


class ImageMultiHash(object):
	def __init__(self, hashes):
		self.segment_hashes = hashes

	def __eq__(self, other):
		if other is None:
			return False
		return self.matches(other)

	def __ne__(self, other):
		return not self.matches(other)

	def __sub__(self, other, hamming_cutoff=None, bit_error_rate=None):
		matches, sum_distance = self.hash_diff(other, hamming_cutoff, bit_error_rate)
		max_difference = len(self.segment_hashes)
		if matches == 0:
			return max_difference
		max_distance = matches * len(self.segment_hashes[0])
		tie_breaker = 0 - (float(sum_distance) / max_distance)
		match_score = matches + tie_breaker
		return max_difference - match_score

	def __hash__(self):
		return hash(tuple(hash(segment) for segment in self.segment_hashes))

	def __str__(self):
		return ",".join(str(x) for x in self.segment_hashes)

	def __repr__(self):
		return repr(self.segment_hashes)

	def hash_diff(self, other_hash, hamming_cutoff=None, bit_error_rate=None):
		# Set default hamming cutoff if it's not set.
		if hamming_cutoff is None and bit_error_rate is None:
			bit_error_rate = 0.25
		if hamming_cutoff is None:
			hamming_cutoff = len(self.segment_hashes[0]) * bit_error_rate
		# Get the hash distance for each region hash within cutoff
		distances = []
		for segment_hash in self.segment_hashes:
			lowest_distance = min(
				segment_hash - other_segment_hash
				for other_segment_hash in other_hash.segment_hashes
			)
			if lowest_distance > hamming_cutoff:
				continue
			distances.append(lowest_distance)
		return len(distances), sum(distances)

	def matches(self, other_hash, region_cutoff=1, hamming_cutoff=None, bit_error_rate=None):

		matches, _ = self.hash_diff(other_hash, hamming_cutoff, bit_error_rate)
		return matches >= region_cutoff

	def best_match(self, other_hashes, hamming_cutoff=None, bit_error_rate=None):

		return min(
			other_hashes,
			key=lambda other_hash: self.__sub__(other_hash, hamming_cutoff, bit_error_rate)
		)


def _find_region(remaining_pixels, segmented_pixels):

	in_region = set()
	not_in_region = set()
	# Find the first pixel in remaining_pixels with a value of True
	available_pixels = numpy.transpose(numpy.nonzero(remaining_pixels))
	start = tuple(available_pixels[0])
	in_region.add(start)
	new_pixels = in_region.copy()
	while True:
		try_next = set()
		# Find surrounding pixels
		for pixel in new_pixels:
			x, y = pixel
			neighbours = [
				(x-1, y),
				(x+1, y),
				(x, y-1),
				(x, y+1)
			]
			try_next.update(neighbours)
		# Remove pixels we have already seen
		try_next.difference_update(segmented_pixels, not_in_region)
		# If there's no more pixels to try, the region is complete
		if not try_next:
			break
		# Empty new pixels set, so we know whose neighbour's to check next time
		new_pixels = set()
		# Check new pixels
		for pixel in try_next:
			if remaining_pixels[pixel]:
				in_region.add(pixel)
				new_pixels.add(pixel)
				segmented_pixels.add(pixel)
			else:
				not_in_region.add(pixel)
	return in_region


def _find_all_segments(pixels, segment_threshold, min_segment_size):

	img_width, img_height = pixels.shape
	# threshold pixels
	threshold_pixels = pixels > segment_threshold
	unassigned_pixels = numpy.full(pixels.shape, True, dtype=bool)

	segments = []
	already_segmented = set()

	# Add all the pixels around the border outside the image:
	already_segmented.update([(-1, z) for z in range(img_height)])
	already_segmented.update([(z, -1) for z in range(img_width)])
	already_segmented.update([(img_width, z) for z in range(img_height)])
	already_segmented.update([(z, img_height) for z in range(img_width)])

	# Find all the "hill" regions
	while numpy.bitwise_and(threshold_pixels, unassigned_pixels).any():
		remaining_pixels = numpy.bitwise_and(threshold_pixels, unassigned_pixels)
		segment = _find_region(remaining_pixels, already_segmented)
		# Apply segment
		if len(segment) > min_segment_size:
			segments.append(segment)
		for pix in segment:
			unassigned_pixels[pix] = False

	# Invert the threshold matrix, and find "valleys"
	threshold_pixels_i = numpy.invert(threshold_pixels)
	while len(already_segmented) < img_width * img_height:
		remaining_pixels = numpy.bitwise_and(threshold_pixels_i, unassigned_pixels)
		segment = _find_region(remaining_pixels, already_segmented)
		# Apply segment
		if len(segment) > min_segment_size:
			segments.append(segment)
		for pix in segment:
			unassigned_pixels[pix] = False

	return segments


def crop_resistant_hash(
		image,
		hash_func=None,
		limit_segments=None,
		segment_threshold=128,
		min_segment_size=500,
		segmentation_image_size=300
	):

	if hash_func is None:
		hash_func = dhash

	orig_image = image.copy()
	# Convert to gray scale and resize
	image = image.convert("L").resize((segmentation_image_size, segmentation_image_size), Image.ANTIALIAS)
	# Add filters
	image = image.filter(ImageFilter.GaussianBlur()).filter(ImageFilter.MedianFilter())
	pixels = numpy.array(image).astype(numpy.float32)

	segments = _find_all_segments(pixels, segment_threshold, min_segment_size)

	# If there are no segments, have 1 segment including the whole image
	if not segments:
		full_image_segment = {(0, 0), (segmentation_image_size-1, segmentation_image_size-1)}
		segments.append(full_image_segment)

	# If segment limit is set, discard the smaller segments
	if limit_segments:
		segments = sorted(segments, key=lambda s: len(s), reverse=True)[:limit_segments]

	# Create bounding box for each segment
	hashes = []
	for segment in segments:
		orig_w, orig_h = orig_image.size
		scale_w = float(orig_w) / segmentation_image_size
		scale_h = float(orig_h) / segmentation_image_size
		min_y = min(coord[0] for coord in segment) * scale_h
		min_x = min(coord[1] for coord in segment) * scale_w
		max_y = (max(coord[0] for coord in segment)+1) * scale_h
		max_x = (max(coord[1] for coord in segment)+1) * scale_w
		# Compute robust hash for each bounding box
		bounding_box = orig_image.crop((min_x, min_y, max_x, max_y))
		hashes.append(hash_func(bounding_box))
		# Show bounding box
		# im_segment = image.copy()
		# for pix in segment:
		# 	im_segment.putpixel(pix[::-1], 255)
		# im_segment.show()
		# bounding_box.show()

	return ImageMultiHash(hashes)
