#version 450


layout(push_constant) uniform PushConstants {
    int iTime;
} pc;

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

void main() {

    // speed factor
    float speed = 0.0005;


    vec2 C = gl_FragCoord.xy;

    float i = 0.0;
    float d;
    float z = fract(dot(C, sin(C))) - 0.5;

    vec4 o = vec4(0.0); // Accumulated color/lighting
    vec4 p;             // Current 3D position along ray
    vec4 O;             // Temporary variable for lighting

    vec2 r = vec2(1920, 1080);

    for (; ++i < 77.0; z += 0.6 * d) {
        // Convert 2D pixel to 3D ray direction
        p = vec4(z * normalize(vec3(C - 0.5 * r, r.y)), 0.1 * pc.iTime * speed);

        // Move through 3D space over time
        p.z += pc.iTime * speed;

        // Save position for lighting calculations
        O = p;

        // Apply rotation matrices to create fractal patterns
        p.xy *= mat2(cos(2.0 + O.z + vec4(0.0, 11.0, 33.0, 0.0)));
        p.xy *= mat2(cos(O + vec4(0.0, 11.0, 33.0, 0.0)));

        // Calculate color based on position and space distortion
        O = (1.0 + sin(0.5 * O.z + length(p - O) + vec4(0.0, 4.0, 3.0, 6.0)))
            / (0.5 + 2.0 * dot(O.xy, O.xy));

        // Domain repetition
        p = abs(fract(p) - 0.5);

        // Distance to nearest surface
        d = abs(min(length(p.xy) - 0.125, min(p.x, p.y) + 1e-3)) + 1e-3;

        // Add lighting contribution
        o += O.w / d * O;
    }

    // Compress brightness to 0-1 range (tone mapping)
    outColor = tanh(o / 2e4)/* * vec4(fragColor, 1)*/;
}