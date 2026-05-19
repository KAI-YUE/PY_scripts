#!/usr/bin/env bash
set -euo pipefail

FF="tools/ffmpeg"
PD="tools/psmfdump"
FFFLAGS=(-hide_banner -loglevel error -y)

pmf2mp4() {
	local in="$1"
	local base name

	echo "Processing file $in..."

	rm -rf "obj"
	mkdir -p "obj" "output/mp4"

	base="$(basename "$in")"
	name="${base%.*}"

	"$PD" "$in" -a "obj/${name}.oma" -v "obj/${name}.264"
	"$FF" "${FFFLAGS[@]}" -i "obj/${name}.264" -i "obj/${name}.oma" -map 0 -map 1 -s 480x272 "output/mp4/${name}.mp4"
}

if [[ -d "input/pmf" ]]; then
	found=0
	while IFS= read -r -d '' f; do
		found=1
		pmf2mp4 "$f"
	done < <(find "input/pmf" -maxdepth 1 -type f -name "*.pmf" -print0)

	if [[ "$found" -eq 0 ]]; then
		echo "No .pmf files found in input/pmf"
	fi
else
	echo "Please put your *.pmf files into input/pmf"
fi

rm -rf "obj"

