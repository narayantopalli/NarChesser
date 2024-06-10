<!DOCTYPE html>
<html lang="en">
<head>
    <style>
        .title {
            display: inline-block;
            position: relative;
        }
        .title::after {
            content: "";
            display: block;
            width: 100%;
            height: 2px;
            background-color: black;
            position: absolute;
            left: 0;
            bottom: -5px; /* Adjust the position as needed */
        }
    </style>
</head>
<body>
    <h1 class="title">Title Text</h1>
</body>
</html>
