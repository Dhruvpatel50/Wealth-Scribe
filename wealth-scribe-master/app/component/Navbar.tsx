'use client'
import React, { useState } from 'react';
import { FaBars } from 'react-icons/fa';
import {
    SignInButton,
    SignedIn,
    SignedOut,
    UserButton,
} from '@clerk/nextjs'
import Link from 'next/link';

const Navbar = () => {
    const [isMenuOpen, setIsMenuOpen] = useState(false);
    const [isSidebarVisible, setIsSidebarVisible] = useState(false);

    const toggleMenu = () => {
        setIsMenuOpen(!isMenuOpen);
    };

    const toggleSidebar = () => {
        setIsSidebarVisible(!isSidebarVisible);
    };

    return (
        <div className="flex">
            <div
                className={`fixed top-0 left-0 z-30 h-full bg-[#171717] text-white transition-transform duration-300 ${isSidebarVisible ? 'translate-x-0' : '-translate-x-full'}`}
                style={{ width: '240px' }}
            >
                <div className="flex justify-between items-center p-4 border-b border-gray-700">
                    <h2 className="font-semibold">Wealth-Scribe</h2>
                    <button
                        className="text-white text-2xl"
                        onClick={toggleSidebar}
                    >
                        &times;
                    </button>
                </div>
                <ul className="mt-4 space-y-4 px-4">
                    <li>
                        <Link href={'/'} className="hover:text-blue-400 transition-colors">Home</Link>
                    </li>
                    <li>
                        <Link href={"/dashboard"} className="hover:text-blue-400 transition-colors">Dashboard</Link>
                    </li>
                </ul>
            </div>

            <div className="flex-1">
                <nav className="flex justify-between bg-[#212121] items-center gap-4 p-4 shadow-lg rounded-lg">
                    <div className="flex items-center">
                        <button
                            type="button"
                            className="btn btn-text btn-circle text-base-content hover:bg-base-content/10 focus:outline-none z-20"
                            aria-label="Toggle Sidebar"
                            onClick={toggleSidebar}
                        >
                            <FaBars className="text-2xl" />
                        </button>
                    </div>
                    <div className="flex items-center">
                        <a className="text-xl font-semibold text-base-content no-underline" href="#">
                            Wealth-Scribe
                        </a>
                    </div>
                    <div className="flex items-center gap-4">
                        <div className="relative inline-flex">
                            <SignedOut>
                                <SignInButton />
                            </SignedOut>
                            <SignedIn>
                                <UserButton />
                            </SignedIn>
                        </div>
                    </div>
                </nav>
            </div>

            {/* Mobile menu toggle */}
            <div className="md:hidden">
                <button onClick={toggleMenu} className="text-2xl text-white">
                    {isMenuOpen ? 'Close Menu' : 'Open Menu'}
                </button>
                {isMenuOpen && (
                    <div className="bg-gray-800 text-white">
                        <ul className="space-y-4 px-4 py-4">
                            <li>
                                <Link href={'/'} className="hover:text-blue-400 transition-colors">Home</Link>
                            </li>
                            <li>
                                <Link href={'/dashboard'} className="hover:text-blue-400 transition-colors">Dashboard</Link>
                            </li>
                            <li>
                                <a href="#" className="hover:text-blue-400 transition-colors">Settings</a>
                            </li>
                            <li>
                                <a href="#" className="hover:text-blue-400 transition-colors">Messages</a>
                            </li>
                            <li>
                                <a href="#" className="hover:text-blue-400 transition-colors">Profile</a>
                            </li>
                        </ul>
                    </div>
                )}
            </div>
        </div>
    );
};

export default Navbar;
