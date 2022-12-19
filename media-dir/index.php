<?php
// Image gallery code:
$images = glob("media-dir/*.{jpg,png,gif}", GLOB_BRACE);
foreach($images as $image)
{
    echo '<img src="'.$image.'"/>';
}
?>