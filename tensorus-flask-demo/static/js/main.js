(function() {
    function applyTheme() {
        if (localStorage.getItem('theme') === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
            document.documentElement.classList.add('dark');
        } else {
            document.documentElement.classList.remove('dark');
        }
    }
    applyTheme(); // Apply theme on initial load

    // Optional: Listen for changes in OS theme preference if no theme is stored in localStorage
    // This is more complex if you also want a manual toggle later, as they can conflict.
    // For now, just initial load based on localStorage or prefers-color-scheme.
    // window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
    //     if (!('theme' in localStorage)) { // Only if no manual override
    //         if (e.matches) {
    //             document.documentElement.classList.add('dark');
    //         } else {
    //             document.documentElement.classList.remove('dark');
    //         }
    //     }
    // });
})();

document.addEventListener('DOMContentLoaded', function () {
    const menuButton = document.getElementById('mobile-menu-button');
    const mobileMenu = document.getElementById('mobile-menu-content'); // Changed ID for clarity

    if (menuButton && mobileMenu) {
        menuButton.addEventListener('click', function () {
            mobileMenu.classList.toggle('hidden');

            // Toggle SVG icons visibility
            const openIcon = document.getElementById('menu-open-icon');
            const closeIcon = document.getElementById('menu-close-icon');
            if (openIcon && closeIcon) {
                openIcon.classList.toggle('hidden');
                closeIcon.classList.toggle('hidden');
            }
        });
    }

    // Optional: Close mobile menu when a link is clicked (if it's a single-page app style nav)
    // const navLinks = mobileMenu.querySelectorAll('a');
    // navLinks.forEach(link => {
    //     link.addEventListener('click', () => {
    //         if (!mobileMenu.classList.contains('hidden')) {
    //             mobileMenu.classList.add('hidden');
    //             if (openIcon && closeIcon) {
    //                 openIcon.classList.remove('hidden');
    //                 closeIcon.classList.add('hidden');
    //             }
    //         }
    //     });
    // });
});
